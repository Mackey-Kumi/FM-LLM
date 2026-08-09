"""
FM-LLM: real test-set evaluation of the trained ETTh1-96 checkpoint.

Everything up to this point has only measured our custom hybrid training/val
loss -- never standard MSE/MAE (paper Eq. 25-26) on the held-out test split.
This script loads the checkpoint from the full training run (best epoch 13,
val_loss=1.0228), runs it over the untouched test set in eval mode, and
computes plain MSE/MAE for direct comparison against the paper's reported
ETTh1-96 numbers (Table A.12: MSE=0.342, MAE=0.380).

Inference only (no gradients), so a single GPU is enough -- no DataParallel
needed here, unlike the training run.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import LlamaModel

torch.manual_seed(0)

assert torch.cuda.is_available(), "No GPU attached -- check the notebook's accelerator setting"
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")

# ---------------------------------------------------------------------------
# 1. Data: same chronological split as training -- this time we need the
#    TEST split, which has never been touched by training or validation.
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

train_data = numeric_data.iloc[0:train_end]  # only needed to compute normalization stats
test_data = numeric_data.iloc[val_end - seq_len : test_end]

channel_mean = train_data.mean(axis=0)
channel_std = train_data.std(axis=0)


def normalize(df):
    return (df - channel_mean) / channel_std


test_norm = normalize(test_data).values.astype("float32")


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


BATCH_SIZE = 256

test_dataset = ForecastWindowDataset(test_norm, seq_len, pred_len)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Test examples: {len(test_dataset)}  |  test batches: {len(test_loader)}")

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
# 3. Frozen Llama-3.2-1B backbone (HF token via an attached private dataset)
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
# 5. Build models and load the trained checkpoint
# ---------------------------------------------------------------------------
embedding_module = FourierEmbeddingModule(patch_len=pred_len, mlp_dim=512, llm_dim=2048).to(device)
llama_backbone = RealLlamaBackbone().to(device)
decoder = FANMoEDecoder(
    llm_dim=2048, hidden_dim=512, patch_len=pred_len,
    num_shared=2, num_routed=2, top_k=1,
).to(device).to(torch.bfloat16)

checkpoint_path = find_input_file("etth1_96_best.pt", "fm-llm-etth1-checkpoint")
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
embedding_module.load_state_dict(checkpoint["embedding_module"])
decoder.load_state_dict(checkpoint["decoder"])
print(f"Checkpoint loaded: epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

embedding_module.eval()
decoder.eval()


def forward_pass(history_batch):
    tokens = history_batch.permute(0, 2, 1).unfold(-1, pred_len, pred_len).to(device)
    embeds = embedding_module(tokens)
    llm_out = llama_backbone(embeds)
    forecast, top_indices, scores = decoder(llm_out)
    pred_patch = forecast[:, :, -1, :]
    return pred_patch


# ---------------------------------------------------------------------------
# 6. Real test-set evaluation: plain MSE/MAE (paper Eq. 25-26), computed on
#    the normalized data -- standard convention for these benchmarks.
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print(f"TEST EVALUATION: ETTh1, input-672-predict-96")
print(f"{'=' * 70}\n")

all_preds = []
all_targets = []

start = time.time()
with torch.no_grad():
    for i, (history_batch, future_batch) in enumerate(test_loader):
        pred_patch = forward_pass(history_batch)
        target_patch = future_batch.permute(0, 2, 1).to(device)

        all_preds.append(pred_patch.float().cpu())
        all_targets.append(target_patch.float().cpu())

        print(f"  batch {i + 1}/{len(test_loader)}")

all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

mse = ((all_preds - all_targets) ** 2).mean().item()
mae = (all_preds - all_targets).abs().mean().item()

print(f"\nEvaluated {all_preds.shape[0]} test windows in {time.time() - start:.1f}s")
print(f"\n{'=' * 70}")
print(f"RESULT: ETTh1-96 test set -- MSE={mse:.4f}  MAE={mae:.4f}")
print(f"Paper (Table A.12, ETTh1-96): MSE=0.342  MAE=0.380")
print(f"{'=' * 70}")

# ---------------------------------------------------------------------------
# 7. Save a few sample windows (all channels) for a predicted-vs-ground-truth
#    plot -- OT (index 6) is the standard visualized target for ETT datasets.
# ---------------------------------------------------------------------------
import numpy as np

sample_indices = [0, len(all_preds) // 2, len(all_preds) - 1]
np.savez(
    "/kaggle/working/sample_predictions.npz",
    pred=all_preds[sample_indices].numpy(),      # (3, channels, pred_len)
    target=all_targets[sample_indices].numpy(),  # (3, channels, pred_len)
    sample_indices=np.array(sample_indices),
    channel_names=np.array(["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]),
)
print("Saved sample_predictions.npz for visualization")
