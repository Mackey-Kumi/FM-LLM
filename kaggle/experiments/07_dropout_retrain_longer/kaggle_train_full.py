"""
FM-LLM: ETTh1 dropout retrain with a higher epoch ceiling than
05_dropout_retrain. That run hit its NUM_EPOCHS=30 ceiling still trending
down (epoch 26-30 val_loss: 0.9773, 0.9643, 0.9701, 0.9639, 0.9589 -- noisy
but improving, with the best value landing on the very last allowed epoch),
meaning patience=3 never actually got to trigger -- the run was cut off by
an arbitrary ceiling, not by genuine convergence. This experiment raises
NUM_EPOCHS to 60 so early stopping can fire naturally. Everything else is
byte-identical to 05_dropout_retrain: same dropout placement, same batch=256
T4x2 DataParallel split, same lr=2e-4, same patience=3, same 12/4/4-month
split.

Module-level code below the class/function definitions is guarded by
`if __name__ == "__main__":` so this file can be imported locally without
triggering CUDA asserts, HF downloads, or the training loop -- on Kaggle it
still runs exactly as before, since Kaggle executes this file as the main
script.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import LlamaModel

torch.manual_seed(0)

DROPOUT = 0.2  # Table 2: ETTh1 dropout=0.2

# ---------------------------------------------------------------------------
# 1. Fourier Embedding Module (paper Eq. 7-11), with dropout (Fig. 2)
# ---------------------------------------------------------------------------
class FourierAnalysisNetwork(nn.Module):
    def __init__(self, hidden_dim=512, p_ratio=0.25):
        super().__init__()
        d_pbar = max(1, round(hidden_dim * p_ratio))
        d_p = (hidden_dim - d_pbar) // 2
        d_pbar = hidden_dim - 2 * d_p
        self.w_p = nn.Linear(hidden_dim, d_p, bias=False)
        self.w_pbar = nn.Linear(hidden_dim, d_pbar)

    def forward(self, x):
        periodic = self.w_p(x)
        trend = F.silu(self.w_pbar(x))
        return torch.cat([torch.cos(periodic), torch.sin(periodic), trend], dim=-1)


class FourierEmbeddingModule(nn.Module):
    def __init__(self, patch_len, mlp_dim, llm_dim, dropout=DROPOUT):
        super().__init__()
        self.linear_in = nn.Linear(patch_len, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.fan = FourierAnalysisNetwork(hidden_dim=mlp_dim)
        self.linear_out = nn.Linear(mlp_dim, llm_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fan(x)
        x = torch.tanh(x)
        return self.linear_out(x)


# ---------------------------------------------------------------------------
# 2. Frozen Llama-3.2-1B backbone (HF token via an attached private dataset)
# ---------------------------------------------------------------------------
def get_hf_token():
    candidate_paths = [
        "/kaggle/input/fm-llm-hf-token/hf_token.txt",
        "/kaggle/input/datasets/mackeykumi/fm-llm-hf-token/hf_token.txt",
    ]
    for token_path in candidate_paths:
        if os.path.exists(token_path):
            with open(token_path) as f:
                return f.read().strip()
    raise RuntimeError(
        f"Could not find HF token in any of: {candidate_paths}. Make sure the "
        "mackeykumi/fm-llm-hf-token dataset is attached via dataset_sources "
        "in kernel-metadata.json."
    )


class RealLlamaBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        hf_token = get_hf_token()
        print("Loading meta-llama/Llama-3.2-1B (frozen)...")
        self.llama = LlamaModel.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            token=hf_token,
            torch_dtype=torch.bfloat16,
        )
        for param in self.llama.parameters():
            param.requires_grad = False
        print("Llama-3.2-1B loaded and frozen.")

    def forward(self, embedding_inputs):
        b, s, n, h = embedding_inputs.shape
        flat_inputs = embedding_inputs.reshape(b * s, n, h).to(torch.bfloat16)
        outputs = self.llama(inputs_embeds=flat_inputs)
        return outputs.last_hidden_state.reshape(b, s, n, h)


# ---------------------------------------------------------------------------
# 3. FAN-MoE decoder (paper Eq. 13-18), with dropout in the Fourier expert
# ---------------------------------------------------------------------------
class FANExpert(nn.Module):
    def __init__(self, llm_dim, hidden_dim, patch_len, dropout=DROPOUT):
        super().__init__()
        self.linear_in = nn.Linear(llm_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fan = FourierAnalysisNetwork(hidden_dim=hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        x = torch.tanh(self.linear_in(x))
        x = self.dropout1(x)
        x = self.fan(x)
        x = F.silu(x)
        x = self.dropout2(x)
        return self.linear_out(x)


class RoutedExpert(nn.Module):
    def __init__(self, llm_dim, hidden_dim, patch_len):
        super().__init__()
        self.layer1 = nn.Linear(llm_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        return self.layer2(F.silu(self.layer1(x)))


class MoEGate(nn.Module):
    def __init__(self, llm_dim, num_routed, top_k, bias_update_rate=1e-3):
        super().__init__()
        self.top_k = top_k
        self.num_routed = num_routed
        self.bias_update_rate = bias_update_rate
        self.score_proj = nn.Linear(llm_dim, num_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_routed))

    def forward(self, x):
        scores = F.softmax(self.score_proj(x), dim=-1)
        biased_scores = scores + self.expert_bias
        _, top_indices = torch.topk(biased_scores, self.top_k, dim=-1)
        selected_raw = torch.gather(scores, -1, top_indices)
        gate_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)
        return gate_weights, top_indices, scores

    @torch.no_grad()
    def update_bias(self, top_indices):
        load = torch.zeros(self.num_routed, device=top_indices.device)
        flat = top_indices.reshape(-1)
        load.scatter_add_(0, flat, torch.ones_like(flat, dtype=load.dtype))
        target = load.mean()
        self.expert_bias += self.bias_update_rate * torch.sign(target - load)


class FANMoEDecoder(nn.Module):
    def __init__(self, llm_dim, hidden_dim, patch_len, num_shared, num_routed, top_k, dropout=DROPOUT):
        super().__init__()
        self.shared_experts = nn.ModuleList(
            [FANExpert(llm_dim, hidden_dim, patch_len, dropout=dropout) for _ in range(num_shared)]
        )
        self.routed_experts = nn.ModuleList(
            [RoutedExpert(llm_dim, hidden_dim, patch_len) for _ in range(num_routed)]
        )
        self.gate = MoEGate(llm_dim, num_routed, top_k)

    def forward(self, x):
        shared_out = sum(expert(x) for expert in self.shared_experts)
        gate_weights, top_indices, scores = self.gate(x)
        full_weights = torch.zeros(
            *x.shape[:-1], len(self.routed_experts), device=x.device, dtype=gate_weights.dtype
        )
        full_weights.scatter_(-1, top_indices, gate_weights)
        routed_out = sum(
            full_weights[..., i:i + 1].to(x.dtype) * expert(x)
            for i, expert in enumerate(self.routed_experts)
        )
        return shared_out + routed_out, top_indices, scores


# ---------------------------------------------------------------------------
# 4. Hybrid time-frequency loss + sequence balance loss (paper Eq. 19-23)
# ---------------------------------------------------------------------------
def hybrid_loss(pred, target, alpha=1.0, beta=1.0):
    pred = pred.float()
    target = target.float()
    L = pred.shape[-1]
    l = torch.arange(1, L + 1, device=pred.device, dtype=pred.dtype)
    w = 1.0 / torch.sqrt(l)
    time_term = (w * (pred - target) ** 2).mean()
    pred_fft = torch.fft.rfft(w * pred, dim=-1)
    target_fft = torch.fft.rfft(w * target, dim=-1)
    freq_term = (pred_fft - target_fft).abs().mean()
    return alpha * freq_term + beta * time_term


def sequence_balance_loss(scores, top_indices, num_routed, top_k, gamma=1e-4):
    T = top_indices.shape[:-1].numel()
    one_hot = F.one_hot(top_indices, num_classes=num_routed).sum(dim=-2)
    f = one_hot.reshape(-1, num_routed).sum(dim=0).float() * (num_routed / (top_k * T))
    s_norm = scores / scores.sum(dim=-1, keepdim=True)
    P = s_norm.reshape(-1, num_routed).mean(dim=0)
    return gamma * (f * P).sum()


# ---------------------------------------------------------------------------
# 5. Combined pipeline module + dataset
# ---------------------------------------------------------------------------
class FullPipeline(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.embedding_module = FourierEmbeddingModule(patch_len=pred_len, mlp_dim=512, llm_dim=2048, dropout=DROPOUT)
        self.llama_backbone = RealLlamaBackbone()
        self.decoder = FANMoEDecoder(
            llm_dim=2048, hidden_dim=512, patch_len=pred_len,
            num_shared=2, num_routed=2, top_k=1, dropout=DROPOUT,
        )

    def forward(self, tokens):
        embeds = self.embedding_module(tokens)
        llm_out = self.llama_backbone(embeds)
        forecast, top_indices, scores = self.decoder(llm_out)
        pred_patch = forecast[:, :, -1, :]
        return pred_patch, top_indices, scores


class ForecastWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=672, pred_len=96):
        self.data = torch.from_numpy(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        history = self.data[idx : idx + self.seq_len]
        future = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return history, future


# ---------------------------------------------------------------------------
# 6. Script execution -- guarded so this module can be imported locally
#    without a GPU, HF token, or network access. Kaggle runs this file
#    directly, so __name__ == "__main__" there and everything below still
#    executes exactly as a normal script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assert torch.cuda.is_available(), "No GPU attached -- check the notebook's accelerator setting"
    NUM_GPUS = torch.cuda.device_count()
    print(f"GPU count visible to torch: {NUM_GPUS}")
    for i in range(NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    assert NUM_GPUS >= 2, "This run requires the T4x2 accelerator (2 GPUs) -- only found 1"

    device = torch.device("cuda:0")
    CHECKPOINT_DIR = "/kaggle/working"

    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    raw_dataframe = pd.read_csv(url)
    numeric_data = raw_dataframe.drop(columns=["date"])

    HOURS_PER_MONTH = 30 * 24
    num_train = 12 * HOURS_PER_MONTH
    num_val = 4 * HOURS_PER_MONTH
    num_test = 4 * HOURS_PER_MONTH

    seq_len = 672
    pred_len = 96

    train_end = num_train
    val_end = num_train + num_val

    train_data = numeric_data.iloc[0:train_end]
    val_data = numeric_data.iloc[train_end - seq_len : val_end]

    channel_mean = train_data.mean(axis=0)
    channel_std = train_data.std(axis=0)

    def normalize(df):
        return (df - channel_mean) / channel_std

    train_norm = normalize(train_data).values.astype("float32")
    val_norm = normalize(val_data).values.astype("float32")

    BATCH_SIZE = 256  # Table 2: ETTh1 batch=256 -- split 128+128 across the two T4s

    train_dataset = ForecastWindowDataset(train_norm, seq_len, pred_len)
    val_dataset = ForecastWindowDataset(val_norm, seq_len, pred_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training examples: {len(train_dataset)}  |  batches per epoch at batch={BATCH_SIZE}: {len(train_loader)}")
    print(f"Validation examples: {len(val_dataset)}  |  val batches: {len(val_loader)}")

    base_pipeline = FullPipeline(pred_len=pred_len).to(device)
    base_pipeline.decoder.to(torch.bfloat16)
    pipeline = nn.DataParallel(base_pipeline, device_ids=list(range(NUM_GPUS)))

    trainable_params = list(base_pipeline.embedding_module.parameters()) + list(base_pipeline.decoder.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=2e-4)  # Table 2: ETTh1 lr=2e-4
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    def forward_pass(history_batch):
        tokens = history_batch.permute(0, 2, 1).unfold(-1, pred_len, pred_len).to(device)
        return pipeline(tokens)

    NUM_EPOCHS = 60  # raised from 05's 30 -- that run was still improving when it hit the ceiling
    PATIENCE = 3
    LAMBDA_BALANCE = 1.0  # Table 2 for ETTh1
    LOG_EVERY = 5

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"\n{'=' * 70}")
    print(f"TRAINING (dropout={DROPOUT}): up to {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, patience={PATIENCE}")
    print(f"{'=' * 70}\n")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        base_pipeline.train()
        train_losses = []

        for i, (history_batch, future_batch) in enumerate(train_loader):
            pred_patch, top_indices, scores = forward_pass(history_batch)
            target_patch = future_batch.permute(0, 2, 1).to(device)

            forecast_loss = hybrid_loss(pred_patch, target_patch)
            balance_loss = sequence_balance_loss(scores, top_indices, num_routed=2, top_k=1)
            loss = forecast_loss + LAMBDA_BALANCE * balance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            base_pipeline.decoder.gate.update_bias(top_indices)

            train_losses.append(loss.item())

            if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(train_loader):
                print(f"  epoch {epoch + 1}/{NUM_EPOCHS}  batch {i + 1}/{len(train_loader)}  loss={loss.item():.4f}")

        base_pipeline.eval()
        val_losses = []
        with torch.no_grad():
            for history_batch, future_batch in val_loader:
                pred_patch, top_indices, scores = forward_pass(history_batch)
                target_patch = future_batch.permute(0, 2, 1).to(device)
                val_losses.append(hybrid_loss(pred_patch, target_patch).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  time={epoch_time / 60:.1f}min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "embedding_module": base_pipeline.embedding_module.state_dict(),
                    "decoder": base_pipeline.decoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                f"{CHECKPOINT_DIR}/etth1_96_dropout_longer_best.pt",
            )
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs")
                break

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {CHECKPOINT_DIR}/etth1_96_dropout_longer_best.pt")
    print(f"{'=' * 70}")
