"""
FM-LLM: precompute 720-step forecasts for the GridForecast demo app.

Runs the exact rollout from 10_rollout_eval_instnorm (same model code, same
bf16 backbone/decoder, same instance-norm wrapping) over every test window
for all four checkpoints, then keeps every STRIDE-th window's full 720-step
trajectory (globally normalized) for the app. The app serves these instantly
without Llama, an HF token, checkpoints or network access.

Also re-derives the full-test-set MSE/MAE at 192/336/720 from the same
predictions, as a check that this run reproduces experiment 10.

Output: /kaggle/working/forecasts.npz, laid out as app/inference.py's
load_precomputed expects:
  window_indices             (W,) int64
  <checkpoint label>         (W, 720, 7) float32
Download it to app/precomputed/forecasts.npz.

Module-level code below the class/function definitions is guarded by
`if __name__ == "__main__":` so this file can be imported locally without a
GPU, HF token, or network access.
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
    STRIDE = 24  # keep one window per day for the app

    train_end = num_train
    val_end = num_train + num_val
    test_end = num_train + num_val + num_test

    train_data = numeric_data.iloc[0:train_end]
    test_data = numeric_data.iloc[val_end - SEQ_LEN : test_end]

    channel_mean = train_data.mean(axis=0)
    channel_std = train_data.std(axis=0)
    test_norm = ((test_data - channel_mean) / channel_std).values.astype("float32")

    BATCH_SIZE = 256
    test_dataset = ForecastWindowDataset(test_norm, seq_len=SEQ_LEN, future_len=MAX_HORIZON)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    keep = np.arange(0, len(test_dataset), STRIDE)
    print(f"Test windows: {len(test_dataset)}  |  kept for app: {len(keep)}")

    llama_backbone = RealLlamaBackbone().to(device)

    CHECKPOINT_SPECS = {
        "no_dropout": ("etth1_96_best.pt", False),
        "dropout": ("etth1_96_dropout_best.pt", False),
        "dropout_longer": ("etth1_96_dropout_longer_best.pt", False),
        "instnorm": ("etth1_96_instnorm_best.pt", True),
    }

    def build_and_load(checkpoint_filename):
        embedding_module = FourierEmbeddingModule(patch_len=PATCH_LEN, mlp_dim=512, llm_dim=2048).to(device)
        decoder = FANMoEDecoder(
            llm_dim=2048, hidden_dim=512, patch_len=PATCH_LEN,
            num_shared=2, num_routed=2, top_k=1,
        ).to(device).to(torch.bfloat16)
        checkpoint_path = find_input_file(checkpoint_filename, "fm-llm-etth1-checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        embedding_module.load_state_dict(checkpoint["embedding_module"])
        decoder.load_state_dict(checkpoint["decoder"])
        embedding_module.eval()
        decoder.eval()
        return embedding_module, decoder

    arrays = {"window_indices": keep.astype(np.int64)}

    for label, (filename, use_instance_norm) in CHECKPOINT_SPECS.items():
        embedding_module, decoder = build_and_load(filename)
        print(f"\n{label} (instance_norm={use_instance_norm})")

        all_preds, all_targets = [], []
        start = time.time()
        with torch.no_grad():
            for i, (history_batch, future_batch) in enumerate(test_loader):
                history = history_batch.permute(0, 2, 1).to(device)
                if use_instance_norm:
                    history_input, mean, std = instance_normalize(history)
                else:
                    history_input = history
                history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)

                def predict_fn(context, embedding_module=embedding_module, decoder=decoder):
                    return predict_next_patch(context, embedding_module, llama_backbone, decoder)

                generated = autoregressive_rollout(history_tokens, predict_fn, NUM_ROLLOUT_STEPS)
                pred = truncate_to_horizon(generated, MAX_HORIZON)
                if use_instance_norm:
                    pred = instance_denormalize(pred, mean, std)
                all_preds.append(pred.float().cpu())
                all_targets.append(future_batch.permute(0, 2, 1).float())
                print(f"  batch {i + 1}/{len(test_loader)}")

        preds = torch.cat(all_preds)      # (N, channels, 720)
        targets = torch.cat(all_targets)
        print(f"  {len(preds)} windows in {time.time() - start:.1f}s")
        for h in HORIZONS:
            mse = ((preds[..., :h] - targets[..., :h]) ** 2).mean().item()
            mae = (preds[..., :h] - targets[..., :h]).abs().mean().item()
            print(f"  h={h}: MSE={mse:.4f}  MAE={mae:.4f}")

        arrays[label] = preds[keep].permute(0, 2, 1).numpy().astype(np.float32)  # (W, 720, channels)

    np.savez_compressed("/kaggle/working/forecasts.npz", **arrays)
    print("\nSaved /kaggle/working/forecasts.npz")
