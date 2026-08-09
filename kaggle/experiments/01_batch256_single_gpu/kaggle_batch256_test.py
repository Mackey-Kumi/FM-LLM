"""
FM-LLM: single-experiment test -- does the paper's true ETTh1 batch size (256,
Table 2) fit in GPU memory on Kaggle, the same way it OOM'd on Colab's free T4?

This is a direct port of the verified architecture from the FM-LLM Colab
notebook (sections 1-11): chronological split, Fourier Embedding Module,
frozen Llama-3.2-1B backbone, FAN-MoE decoder, hybrid time-frequency loss.
The only changes from the Colab version: HF token comes from a Kaggle Secret
instead of google.colab.userdata, and there's no Drive mount / checkpointing --
this run's only job is to attempt a few real train steps at batch=256 and
report, unambiguously, whether it fits.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import LlamaModel

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "No GPU attached -- check the notebook's accelerator setting"

gpu_props = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu_props.name}  |  VRAM: {gpu_props.total_memory / 1e9:.1f} GB")
print(f"GPU count visible to torch: {torch.cuda.device_count()}")
print(f"PyTorch: {torch.__version__}")

# ---------------------------------------------------------------------------
# 1. Data: chronological split, per-channel normalization (train stats only)
# ---------------------------------------------------------------------------
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
test_end = num_train + num_val + num_test

train_data = numeric_data.iloc[0:train_end]
val_data = numeric_data.iloc[train_end - seq_len : val_end]

channel_mean = train_data.mean(axis=0)
channel_std = train_data.std(axis=0)


def normalize(df):
    return (df - channel_mean) / channel_std


train_norm = normalize(train_data).values.astype("float32")
val_norm = normalize(val_data).values.astype("float32")


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


BATCH_SIZE = 256  # Table 2: ETTh1 batch=256 -- this is the number under test

train_dataset = ForecastWindowDataset(train_norm, seq_len, pred_len)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Training examples: {len(train_dataset)}  |  batches per epoch at batch={BATCH_SIZE}: {len(train_loader)}")

# ---------------------------------------------------------------------------
# 2. Fourier Embedding Module (paper Eq. 7-11)
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
    def __init__(self, patch_len, mlp_dim, llm_dim):
        super().__init__()
        self.linear_in = nn.Linear(patch_len, mlp_dim)
        self.fan = FourierAnalysisNetwork(hidden_dim=mlp_dim)
        self.linear_out = nn.Linear(mlp_dim, llm_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = torch.tanh(self.fan(F.silu(x)))
        return self.linear_out(x)


# ---------------------------------------------------------------------------
# 3. Frozen Llama-3.2-1B backbone (HF token via an attached private dataset --
#    Kaggle Secrets don't survive CLI-pushed kernel versions, only UI-saved ones)
# ---------------------------------------------------------------------------
def get_hf_token():
    # Kaggle's dataset mount path has been observed to vary between kernels --
    # sometimes /kaggle/input/<dataset>/, sometimes /kaggle/input/datasets/<owner>/<dataset>/
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
        flat_inputs = embedding_inputs.reshape(b * s, n, h).to(self.llama.dtype)
        outputs = self.llama(inputs_embeds=flat_inputs)
        return outputs.last_hidden_state.reshape(b, s, n, h)


# ---------------------------------------------------------------------------
# 4. FAN-MoE decoder (paper Eq. 13-18)
# ---------------------------------------------------------------------------
class FANExpert(nn.Module):
    def __init__(self, llm_dim, hidden_dim, patch_len):
        super().__init__()
        self.linear_in = nn.Linear(llm_dim, hidden_dim)
        self.fan = FourierAnalysisNetwork(hidden_dim=hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        x = torch.tanh(self.linear_in(x))
        x = self.fan(x)
        return self.linear_out(F.silu(x))


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
    def __init__(self, llm_dim, hidden_dim, patch_len, num_shared, num_routed, top_k):
        super().__init__()
        self.shared_experts = nn.ModuleList(
            [FANExpert(llm_dim, hidden_dim, patch_len) for _ in range(num_shared)]
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
# 5. Hybrid time-frequency loss + sequence balance loss (paper Eq. 19-23)
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
# 6. Build the models
# ---------------------------------------------------------------------------
embedding_module = FourierEmbeddingModule(patch_len=pred_len, mlp_dim=512, llm_dim=2048).to(device)
llama_backbone = RealLlamaBackbone().to(device)
decoder = FANMoEDecoder(
    llm_dim=2048, hidden_dim=512, patch_len=pred_len,
    num_shared=2, num_routed=2, top_k=1,
).to(device).to(torch.bfloat16)

trainable_params = list(embedding_module.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=2e-4)
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")


def forward_pass(history_batch):
    tokens = history_batch.permute(0, 2, 1).unfold(-1, pred_len, pred_len).to(device)
    embeds = embedding_module(tokens)
    llm_out = llama_backbone(embeds)
    forecast, top_indices, scores = decoder(llm_out)
    pred_patch = forecast[:, :, -1, :]
    return pred_patch, top_indices, scores


# ---------------------------------------------------------------------------
# 7. THE EXPERIMENT: attempt real train steps at batch=256, report memory
# ---------------------------------------------------------------------------
NUM_TEST_BATCHES = 3  # a few real optimizer steps is enough to prove fit/no-fit

print(f"\n{'=' * 70}")
print(f"EXPERIMENT: {NUM_TEST_BATCHES} real train steps at batch_size={BATCH_SIZE}")
print(f"{'=' * 70}\n")

torch.cuda.reset_peak_memory_stats()
embedding_module.train()
decoder.train()

result = "UNKNOWN"
try:
    for i, (history_batch, future_batch) in enumerate(train_loader):
        if i >= NUM_TEST_BATCHES:
            break

        t0 = time.time()
        pred_patch, top_indices, scores = forward_pass(history_batch)
        target_patch = future_batch.permute(0, 2, 1).to(device)

        forecast_loss = hybrid_loss(pred_patch, target_patch)
        balance_loss = sequence_balance_loss(scores, top_indices, num_routed=2, top_k=1)
        loss = forecast_loss + balance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        decoder.gate.update_bias(top_indices)
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"step {i + 1}/{NUM_TEST_BATCHES}  loss={loss.item():.4f}  "
            f"time={time.time() - t0:.2f}s  allocated={allocated:.2f}GB  "
            f"reserved={reserved:.2f}GB  peak={peak:.2f}GB"
        )

    result = "SUCCESS"

except torch.cuda.OutOfMemoryError as e:
    result = "OOM"
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nCUDA OutOfMemoryError at batch_size={BATCH_SIZE}")
    print(f"Peak memory allocated before failure: {peak:.2f} GB")
    print(f"GPU capacity: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"\nRaw error:\n{e}")

print(f"\n{'=' * 70}")
print(f"VERDICT: batch_size={BATCH_SIZE} on {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB) -> {result}")
if result == "SUCCESS":
    print(f"Peak memory allocated across {NUM_TEST_BATCHES} steps: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"{'=' * 70}")
