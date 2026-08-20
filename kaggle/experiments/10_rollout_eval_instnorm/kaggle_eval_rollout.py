"""
FM-LLM: autoregressive rollout evaluation for ETTh1 horizons 192/336/720,
run against all four checkpoints produced so far:
  - no_dropout:     etth1_96_best.pt                (03_full_train_dual_gpu)
  - dropout:        etth1_96_dropout_best.pt         (05_dropout_retrain, ceiling=30)
  - dropout_longer: etth1_96_dropout_longer_best.pt  (07_dropout_retrain_longer, ceiling=60)
  - instnorm:       etth1_96_instnorm_best.pt        (09_instance_norm_retrain)

Identical rollout mechanics to 08_rollout_eval_longer for the first three
checkpoints. The instnorm checkpoint needs one extra wrapping step, since it
was trained with instance (per-window) normalization on top of the existing
global normalization (see 09_instance_norm_retrain's docstring for the full
rationale): its history window is normalized with its own mean/std before
tokenizing, the rollout runs entirely in that normalized space exactly like
the other checkpoints (so autoregressive_rollout needs zero changes), and
only once a trajectory is truncated to a horizon (batch, channels, horizon)
is it denormalized back with that same window's fixed mean/std -- matching
what the same window's mean/std broadcasts against. That fixed mean/std is
computed once from the original 672-step history and reused for every
rollout step, never recomputed mid-rollout, so the model's own compounding
predictions don't feed back into its normalization statistics.

The trained model only ever predicts one 96-step patch per forward pass
(paper Section 3.2.4/3.3: training loss is computed only on the token
immediately following the context window). Longer horizons come from
feeding the model's own predictions back in as input and repeating --
Section 3.3: "the recursive use of model outputs as future inputs." Context
is a fixed 7-token (672-step) sliding window: each new prediction is
appended and the oldest token is dropped.

336 and 720 aren't multiples of 96 (96x3=288, 96x4=384; 96x7=672, 96x8=768),
so the rollout over-generates 8 patches (768 steps) per window and truncates
to the exact horizon before scoring -- one rollout trajectory serves all
three horizons.

Module-level code below the class/function definitions is guarded by
`if __name__ == "__main__":` so this file can be imported locally without a
GPU, HF token, or network access. Kaggle runs this file directly, so
everything below still executes as a normal script there.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel

torch.manual_seed(0)

DROPOUT = 0.2  # inactive under eval(), kept only so all checkpoints' state_dicts load cleanly

# ---------------------------------------------------------------------------
# 1. Fourier Embedding Module (paper Eq. 7-11) -- identical to
#    09_instance_norm_retrain/kaggle_train_full.py
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
# 2. Frozen Llama-3.2-1B backbone + private-dataset path fallbacks
# ---------------------------------------------------------------------------
def find_input_file(filename, dataset_slug):
    candidate_paths = [
        f"/kaggle/input/{dataset_slug}/{filename}",
        f"/kaggle/input/datasets/mackeykumi/{dataset_slug}/{filename}",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(
        f"Could not find {filename} in any of: {candidate_paths}. Make sure the "
        f"mackeykumi/{dataset_slug} dataset is attached via dataset_sources "
        "in kernel-metadata.json."
    )


def get_hf_token():
    token_path = find_input_file("hf_token.txt", "fm-llm-hf-token")
    with open(token_path) as f:
        return f.read().strip()


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
# 3. FAN-MoE decoder (paper Eq. 13-18) -- identical to
#    09_instance_norm_retrain/kaggle_train_full.py, minus the training-only
#    update_bias method (this script never trains, so the bias buffer is
#    loaded from the checkpoint and never mutated).
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
    def __init__(self, llm_dim, num_routed, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_routed = num_routed
        self.score_proj = nn.Linear(llm_dim, num_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_routed))

    def forward(self, x):
        scores = F.softmax(self.score_proj(x), dim=-1)
        biased_scores = scores + self.expert_bias
        _, top_indices = torch.topk(biased_scores, self.top_k, dim=-1)
        selected_raw = torch.gather(scores, -1, top_indices)
        gate_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)
        return gate_weights, top_indices, scores


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
# 4. Autoregressive rollout logic -- identical to 08_rollout_eval_longer
# ---------------------------------------------------------------------------
def predict_next_patch(tokens, embedding_module, backbone, decoder):
    embeds = embedding_module(tokens)
    llm_out = backbone(embeds)
    forecast, _, _ = decoder(llm_out)
    return forecast[:, :, -1, :]


def autoregressive_rollout(initial_context, predict_fn, num_steps):
    """
    initial_context: (batch, channels, num_tokens, patch_len)
    predict_fn: callable(context) -> next_patch of shape (batch, channels, patch_len)
    Returns: (batch, channels, num_steps, patch_len), the generated patches in call order.
    """
    context = initial_context
    generated = []
    for _ in range(num_steps):
        next_patch = predict_fn(context)
        generated.append(next_patch)
        context = torch.cat([context[:, :, 1:, :], next_patch.unsqueeze(2)], dim=2)
    return torch.stack(generated, dim=2)


def truncate_to_horizon(generated_patches, horizon):
    """
    generated_patches: (batch, channels, num_steps, patch_len)
    horizon: target number of time steps, must be <= num_steps * patch_len
    Returns: (batch, channels, horizon)
    """
    batch, channels, num_steps, patch_len = generated_patches.shape
    flat = generated_patches.reshape(batch, channels, num_steps * patch_len)
    return flat[:, :, :horizon]


# ---------------------------------------------------------------------------
# 5. Instance (per-window) normalization -- identical to
#    09_instance_norm_retrain/kaggle_train_full.py
# ---------------------------------------------------------------------------
def instance_normalize(history):
    """
    history: (batch, channels, seq_len) -- already globally-normalized values
    Returns: (history_norm, mean, std), mean/std shape (batch, channels, 1).
    """
    mean = history.mean(dim=-1, keepdim=True)
    std = history.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
    history_norm = (history - mean) / std
    return history_norm, mean, std


def instance_denormalize(x, mean, std):
    """x: (..., L) with the same leading (batch, channels) dims as mean/std."""
    return x * std + mean


# ---------------------------------------------------------------------------
# 6. Test-set windowing, generalized for the max horizon (720) needed here
# ---------------------------------------------------------------------------
class ForecastWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=672, future_len=720):
        self.data = torch.from_numpy(data)
        self.seq_len = seq_len
        self.future_len = future_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.future_len + 1

    def __getitem__(self, idx):
        history = self.data[idx : idx + self.seq_len]
        future = self.data[idx + self.seq_len : idx + self.seq_len + self.future_len]
        return history, future


# ---------------------------------------------------------------------------
# 7. Script execution -- guarded so this module can be imported locally
#    without a GPU, HF token, or network access. Kaggle runs this file
#    directly, so everything below still executes as a normal script there.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    assert torch.cuda.is_available(), "No GPU attached -- check the notebook's accelerator setting"
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    raw_dataframe = pd.read_csv(url)
    numeric_data = raw_dataframe.drop(columns=["date"])

    HOURS_PER_MONTH = 30 * 24
    num_train = 12 * HOURS_PER_MONTH
    num_val = 4 * HOURS_PER_MONTH
    num_test = 4 * HOURS_PER_MONTH

    SEQ_LEN = 672
    PATCH_LEN = 96
    MAX_HORIZON = 720
    HORIZONS = (192, 336, 720)
    NUM_ROLLOUT_STEPS = -(-MAX_HORIZON // PATCH_LEN)  # ceil(720/96) = 8

    train_end = num_train
    val_end = num_train + num_val
    test_end = num_train + num_val + num_test

    train_data = numeric_data.iloc[0:train_end]  # only needed for normalization stats
    test_data = numeric_data.iloc[val_end - SEQ_LEN : test_end]

    channel_mean = train_data.mean(axis=0)
    channel_std = train_data.std(axis=0)

    def normalize(df):
        return (df - channel_mean) / channel_std

    test_norm = normalize(test_data).values.astype("float32")

    BATCH_SIZE = 256
    test_dataset = ForecastWindowDataset(test_norm, seq_len=SEQ_LEN, future_len=MAX_HORIZON)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test examples: {len(test_dataset)}  |  test batches: {len(test_loader)}")

    llama_backbone = RealLlamaBackbone().to(device)

    def build_and_load(checkpoint_filename):
        embedding_module = FourierEmbeddingModule(patch_len=PATCH_LEN, mlp_dim=512, llm_dim=2048).to(device)
        decoder = FANMoEDecoder(
            llm_dim=2048, hidden_dim=512, patch_len=PATCH_LEN,
            num_shared=2, num_routed=2, top_k=1,
        ).to(device).to(torch.bfloat16)

        checkpoint_path = find_input_file(checkpoint_filename, "fm-llm-etth1-checkpoint")
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        embedding_module.load_state_dict(checkpoint["embedding_module"])
        decoder.load_state_dict(checkpoint["decoder"])
        print(f"Checkpoint loaded: epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

        embedding_module.eval()
        decoder.eval()
        return embedding_module, decoder

    # label -> (checkpoint filename, whether this checkpoint needs the
    # instance-normalize/denormalize wrapper around its rollout)
    CHECKPOINT_SPECS = {
        "no_dropout": ("etth1_96_best.pt", False),
        "dropout": ("etth1_96_dropout_best.pt", False),
        "dropout_longer": ("etth1_96_dropout_longer_best.pt", False),
        "instnorm": ("etth1_96_instnorm_best.pt", True),
    }
    CHECKPOINT_LABELS = tuple(CHECKPOINT_SPECS.keys())

    checkpoints = {
        label: (*build_and_load(filename), use_instance_norm)
        for label, (filename, use_instance_norm) in CHECKPOINT_SPECS.items()
    }

    results = {}

    for label, (embedding_module, decoder, use_instance_norm) in checkpoints.items():
        print(f"\n{'=' * 70}")
        print(f"ROLLOUT EVAL: {label}  (instance_norm={use_instance_norm})")
        print(f"{'=' * 70}\n")

        horizon_preds = {h: [] for h in HORIZONS}
        horizon_targets = {h: [] for h in HORIZONS}

        start = time.time()
        with torch.no_grad():
            for i, (history_batch, future_batch) in enumerate(test_loader):
                history = history_batch.permute(0, 2, 1).to(device)  # (batch, channels, seq_len)
                target_full = future_batch.permute(0, 2, 1).to(device)

                if use_instance_norm:
                    history_input, mean, std = instance_normalize(history)
                else:
                    history_input = history

                history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)

                def predict_fn(context, embedding_module=embedding_module, decoder=decoder):
                    return predict_next_patch(context, embedding_module, llama_backbone, decoder)

                generated = autoregressive_rollout(history_tokens, predict_fn, NUM_ROLLOUT_STEPS)

                for h in HORIZONS:
                    pred_h = truncate_to_horizon(generated, h)
                    if use_instance_norm:
                        pred_h = instance_denormalize(pred_h, mean, std)
                    target_h = target_full[:, :, :h]
                    horizon_preds[h].append(pred_h.float().cpu())
                    horizon_targets[h].append(target_h.float().cpu())

                print(f"  batch {i + 1}/{len(test_loader)}")

        elapsed = time.time() - start
        print(f"\nEvaluated {len(test_dataset)} test windows in {elapsed:.1f}s")

        results[label] = {}
        for h in HORIZONS:
            preds = torch.cat(horizon_preds[h], dim=0)
            targets = torch.cat(horizon_targets[h], dim=0)
            mse = ((preds - targets) ** 2).mean().item()
            mae = (preds - targets).abs().mean().item()
            results[label][h] = (mse, mae)

    print(f"\n{'=' * 70}")
    print("RESULT: ETTh1 autoregressive rollout, input=672")
    print(f"{'=' * 70}")
    print(f"{'horizon':>8} {'checkpoint':>15} {'MSE':>10} {'MAE':>10}")
    for h in HORIZONS:
        for label in CHECKPOINT_LABELS:
            mse, mae = results[label][h]
            print(f"{h:>8} {label:>15} {mse:>10.4f} {mae:>10.4f}")
    print(f"{'=' * 70}")

    np.savez(
        "/kaggle/working/rollout_results.npz",
        **{f"{label}_h{h}_mse": results[label][h][0] for label in results for h in HORIZONS},
        **{f"{label}_h{h}_mae": results[label][h][1] for label in results for h in HORIZONS},
    )
    print("Saved rollout_results.npz")
