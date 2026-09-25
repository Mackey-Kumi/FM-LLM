"""
Inference engine for the GridForecast app.

Forecasts come from one of two sources, per (checkpoint, window):

1. Precomputed (`app/precomputed/forecasts.npz`) -- produced offline by
   `kaggle/experiments/11_app_forecasts` (GPU) or `python -m
   app.precompute_forecasts` (CPU). Instant, and needs no Llama download,
   HF token, checkpoint files or network, so the demo can't stall on stage.
2. Live -- the full FM-LLM pipeline (Fourier embedding -> frozen
   Llama-3.2-1B -> FAN-MoE decoder, 8-step autoregressive rollout), exactly
   as in `kaggle/experiments/10_rollout_eval_instnorm`. Needs the checkpoint
   `.pt` files plus Llama access; Llama is loaded once, lazily, and shared by
   every checkpoint (it's frozen, so it's identical across all of them).

All model-side math happens in the globally-normalized space the model was
trained in, and MSE/MAE are reported there too so they're directly
comparable to the paper's Table A.12. Everything returned for display
(history, forecast, actuals) is converted back to real units.
"""

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model constants (matching training)
DROPOUT = 0.2  # inactive under eval(), kept so every checkpoint's state_dict loads
PATCH_LEN = 96
SEQ_LEN = 672  # 7 tokens x 96h = 28 days of hourly history
LLM_DIM = 2048
MLP_DIM = 512
HIDDEN_DIM = 512
NUM_SHARED = 2
NUM_ROUTED = 2
TOP_K = 1

MAX_HORIZON = 720  # 30 days
NUM_ROLLOUT_STEPS = -(-MAX_HORIZON // PATCH_LEN)  # 8
HORIZONS = (96, 192, 336, 720)

APP_DIR = Path(__file__).parent
CHECKPOINT_DIR = APP_DIR.parent / "kaggle" / "datasets" / "checkpoint"
DATA_CSV = APP_DIR / "demo_data" / "etth1_full.csv"
DATA_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
PRECOMPUTED_FILE = APP_DIR / "precomputed" / "forecasts.npz"

# label -> (checkpoint filename, whether it needs the instance-norm wrapper)
CHECKPOINT_SPECS = {
    "instnorm": ("etth1_96_instnorm_best.pt", True),  # Best model
    "dropout_longer": ("etth1_96_dropout_longer_best.pt", False),
    "dropout": ("etth1_96_dropout_best.pt", False),
    "no_dropout": ("etth1_96_best.pt", False),
}
CHECKPOINT_DESCRIPTIONS = {
    "instnorm": "Dropout + per-window instance normalization (best)",
    "dropout_longer": "Dropout, longer training ceiling",
    "dropout": "Dropout (Table 2: 0.2)",
    "no_dropout": "Baseline, no dropout",
}

CHANNEL_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
CHANNEL_DESCRIPTIONS = {
    "HUFL": "High useful load",
    "HULL": "High useless load",
    "MUFL": "Middle useful load",
    "MULL": "Middle useless load",
    "LUFL": "Low useful load",
    "LULL": "Low useless load",
    "OT": "Oil temperature (target)",
}

# Full-test-set rollout results (all 2161 windows), from
# kaggle/experiments/10_rollout_eval_instnorm. (MSE, MAE), globally normalized.
REPORTED_RESULTS = {
    "no_dropout": {192: (0.5089, 0.4882), 336: (0.5423, 0.5158), 720: (0.6381, 0.5792)},
    "dropout": {192: (0.4417, 0.4571), 336: (0.4858, 0.4925), 720: (0.5894, 0.5661)},
    "dropout_longer": {192: (0.4292, 0.4445), 336: (0.4641, 0.4730), 720: (0.5422, 0.5350)},
    "instnorm": {192: (0.3945, 0.4225), 336: (0.4149, 0.4402), 720: (0.4721, 0.4839)},
}
PAPER_RESULTS = {192: (0.377, 0.403), 336: (0.395, 0.415), 720: (0.397, 0.429)}


# ---------------------------------------------------------------------------
# Model (identical to kaggle/experiments/10_rollout_eval_instnorm)
# ---------------------------------------------------------------------------
class FourierAnalysisNetwork(nn.Module):
    def __init__(self, hidden_dim=MLP_DIM, p_ratio=0.25):
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
    def __init__(self, patch_len=PATCH_LEN, mlp_dim=MLP_DIM, llm_dim=LLM_DIM, dropout=DROPOUT):
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


class FANExpert(nn.Module):
    def __init__(self, llm_dim=LLM_DIM, hidden_dim=HIDDEN_DIM, patch_len=PATCH_LEN, dropout=DROPOUT):
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
    def __init__(self, llm_dim=LLM_DIM, hidden_dim=HIDDEN_DIM, patch_len=PATCH_LEN):
        super().__init__()
        self.layer1 = nn.Linear(llm_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        return self.layer2(F.silu(self.layer1(x)))


class MoEGate(nn.Module):
    def __init__(self, llm_dim=LLM_DIM, num_routed=NUM_ROUTED, top_k=TOP_K):
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
    def __init__(self, llm_dim=LLM_DIM, hidden_dim=HIDDEN_DIM, patch_len=PATCH_LEN,
                 num_shared=NUM_SHARED, num_routed=NUM_ROUTED, top_k=TOP_K, dropout=DROPOUT):
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


def _hf_token() -> Optional[str]:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


class LlamaBackbone(nn.Module):
    """Frozen Llama-3.2-1B. bfloat16 on GPU (as in training), float32 on CPU."""

    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import LlamaModel

        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        print(f"Loading meta-llama/Llama-3.2-1B ({dtype}) on {device}...")
        self.dtype = dtype
        self.llama = LlamaModel.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=_hf_token(),
        )
        for param in self.llama.parameters():
            param.requires_grad = False
        self.llama.to(device)
        self.llama.eval()
        print("Llama-3.2-1B loaded.")

    def forward(self, embedding_inputs):
        b, s, n, h = embedding_inputs.shape
        flat_inputs = embedding_inputs.reshape(b * s, n, h).to(self.dtype)
        with torch.no_grad():
            outputs = self.llama(inputs_embeds=flat_inputs)
        return outputs.last_hidden_state.reshape(b, s, n, h).float()


# ---------------------------------------------------------------------------
# Rollout (identical mechanics to kaggle/experiments/10_rollout_eval_instnorm)
# ---------------------------------------------------------------------------
def instance_normalize(history):
    mean = history.mean(dim=-1, keepdim=True)
    std = history.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
    history_norm = (history - mean) / std
    return history_norm, mean, std


def instance_denormalize(x, mean, std):
    return x * std + mean


def predict_next_patch(tokens, embedding_module, backbone, decoder):
    embeds = embedding_module(tokens)
    llm_out = backbone(embeds)
    forecast, _, _ = decoder(llm_out)
    return forecast[:, :, -1, :]


def autoregressive_rollout(initial_context, predict_fn, num_steps):
    context = initial_context
    generated = []
    for _ in range(num_steps):
        next_patch = predict_fn(context)
        generated.append(next_patch)
        context = torch.cat([context[:, :, 1:, :], next_patch.unsqueeze(2)], dim=2)
    return torch.stack(generated, dim=2)


def truncate_to_horizon(generated_patches, horizon):
    batch, channels, num_steps, patch_len = generated_patches.shape
    flat = generated_patches.reshape(batch, channels, num_steps * patch_len)
    return flat[:, :, :horizon]


def rollout_forecast(history, embedding_module, decoder, backbone, use_instance_norm):
    """
    history: (batch, channels, SEQ_LEN), globally normalized.
    Returns: (batch, channels, MAX_HORIZON), globally normalized.
    """
    if use_instance_norm:
        history_input, mean, std = instance_normalize(history)
    else:
        history_input = history

    history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)

    def predict_fn(context):
        return predict_next_patch(context, embedding_module, backbone, decoder)

    with torch.no_grad():
        generated = autoregressive_rollout(history_tokens, predict_fn, NUM_ROLLOUT_STEPS)
        pred = truncate_to_horizon(generated, MAX_HORIZON)
        if use_instance_norm:
            pred = instance_denormalize(pred, mean, std)
    return pred


def load_checkpoint(checkpoint_name, device):
    filename, use_instance_norm = CHECKPOINT_SPECS[checkpoint_name]
    checkpoint_path = CHECKPOINT_DIR / filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    embedding_module = FourierEmbeddingModule().to(device)
    decoder = FANMoEDecoder().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    embedding_module.load_state_dict(checkpoint["embedding_module"])
    decoder.load_state_dict(checkpoint["decoder"])

    embedding_module.eval()
    decoder.eval()
    return embedding_module, decoder, use_instance_norm


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ETTh1Data:
    """ETTh1 with the paper's 12/4/4-month split and train-set normalization."""

    def __init__(self, csv_path: Path = DATA_CSV):
        if csv_path.exists():
            raw_dataframe = pd.read_csv(csv_path)
        else:
            raw_dataframe = pd.read_csv(DATA_URL)

        hours_per_month = 30 * 24
        num_train = 12 * hours_per_month
        num_val = 4 * hours_per_month
        num_test = 4 * hours_per_month
        val_end = num_train + num_val
        test_end = val_end + num_test

        numeric_data = raw_dataframe.drop(columns=["date"])
        train_data = numeric_data.iloc[:num_train]
        test_slice = slice(val_end - SEQ_LEN, test_end)

        self.channel_names: List[str] = numeric_data.columns.tolist()
        self.channel_mean = train_data.mean(axis=0).values.astype("float32")
        self.channel_std = train_data.std(axis=0).values.astype("float32")
        self.test_raw = numeric_data.iloc[test_slice].values.astype("float32")
        self.test_norm = ((self.test_raw - self.channel_mean) / self.channel_std).astype("float32")
        self.test_dates = pd.to_datetime(raw_dataframe["date"].iloc[test_slice]).reset_index(drop=True)
        self.num_windows = len(self.test_norm) - SEQ_LEN - MAX_HORIZON + 1

    def normalize(self, raw):
        return (raw - self.channel_mean) / self.channel_std

    def denormalize(self, norm):
        return norm * self.channel_std + self.channel_mean


def load_precomputed(path: Path = PRECOMPUTED_FILE) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Returns {checkpoint: {window_idx: (MAX_HORIZON, channels) normalized array}}.
    File layout: `window_indices` (W,) plus one (W, MAX_HORIZON, channels)
    array per checkpoint label.
    """
    if not path.exists():
        return {}
    data = np.load(path)
    windows = data["window_indices"].tolist()
    return {
        name: {w: data[name][i] for i, w in enumerate(windows)}
        for name in CHECKPOINT_SPECS
        if name in data.files
    }


def _metrics(pred_norm, truth_norm):
    err = pred_norm - truth_norm
    return float((err ** 2).mean()), float(np.abs(err).mean())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ForecastEngine:
    """Main inference engine for the GridForecast application."""

    def __init__(self, device="cpu", backbone: Optional[nn.Module] = None,
                 precomputed_path: Path = PRECOMPUTED_FILE):
        self.device = device
        self.data = ETTh1Data()
        self.channel_names = self.data.channel_names
        self.precomputed = load_precomputed(precomputed_path)
        self._backbone = backbone
        self._models: Dict[str, tuple] = {}
        self._live_cache: Dict[tuple, np.ndarray] = {}
        self._lock = threading.Lock()
        print(
            f"ForecastEngine ready: {self.data.num_windows} test windows, "
            f"precomputed={ {k: len(v) for k, v in self.precomputed.items()} }, "
            f"live checkpoints={self.live_checkpoints()}"
        )

    # -- availability -----------------------------------------------------
    def live_checkpoints(self) -> List[str]:
        return [n for n, (f, _) in CHECKPOINT_SPECS.items() if (CHECKPOINT_DIR / f).exists()]

    def available_checkpoints(self) -> List[str]:
        live = set(self.live_checkpoints())
        return [n for n in CHECKPOINT_SPECS if n in live or n in self.precomputed]

    def is_live(self, checkpoint: str) -> bool:
        return checkpoint in self.live_checkpoints()

    def windows(self, checkpoint: str) -> List[int]:
        if self.is_live(checkpoint):
            return list(range(self.data.num_windows))
        return sorted(self.precomputed.get(checkpoint, {}))

    def window_start(self, window_idx: int) -> pd.Timestamp:
        return self.data.test_dates[window_idx + SEQ_LEN]

    def get_channel_names(self) -> List[str]:
        return self.channel_names

    # -- model ------------------------------------------------------------
    def _get_backbone(self):
        if self._backbone is None:
            try:
                self._backbone = LlamaBackbone(self.device)
            except Exception as e:
                raise RuntimeError(
                    "Could not load meta-llama/Llama-3.2-1B for live inference "
                    f"({e}). Set HF_TOKEN in .env, or use precomputed forecasts."
                ) from e
        return self._backbone

    def _get_model(self, checkpoint: str):
        if checkpoint not in self._models:
            self._models[checkpoint] = load_checkpoint(checkpoint, self.device)
        return self._models[checkpoint]

    def _live_rollout(self, checkpoint: str, history_norm: np.ndarray) -> np.ndarray:
        """history_norm: (SEQ_LEN, channels) -> (MAX_HORIZON, channels), normalized."""
        if not self.is_live(checkpoint):
            raise RuntimeError(
                f"Live inference for '{checkpoint}' needs {CHECKPOINT_SPECS[checkpoint][0]} "
                f"in {CHECKPOINT_DIR}."
            )
        with self._lock:
            embedding_module, decoder, use_instance_norm = self._get_model(checkpoint)
            backbone = self._get_backbone()
            history = torch.from_numpy(np.ascontiguousarray(history_norm.T)).unsqueeze(0).to(self.device)
            pred = rollout_forecast(history, embedding_module, decoder, backbone, use_instance_norm)
        return pred.squeeze(0).cpu().numpy().T

    def _live_predict(self, checkpoint: str, window_idx: int) -> np.ndarray:
        key = (checkpoint, window_idx)
        if key not in self._live_cache:
            history = self.data.test_norm[window_idx: window_idx + SEQ_LEN]
            self._live_cache[key] = self._live_rollout(checkpoint, history)
        return self._live_cache[key]

    def _predict(self, checkpoint: str, window_idx: int):
        """Returns ((MAX_HORIZON, channels) normalized prediction, source)."""
        if not 0 <= window_idx < self.data.num_windows:
            raise ValueError(f"window_idx must be in [0, {self.data.num_windows})")
        pre = self.precomputed.get(checkpoint, {})
        if window_idx in pre:
            return pre[window_idx], "precomputed"
        return self._live_predict(checkpoint, window_idx), "live"

    # -- public API ---------------------------------------------------------
    def forecast(self, window_idx: int, horizon: int, checkpoint: str = "instnorm",
                 context_hours: int = 168) -> Dict:
        pred_norm, source = self._predict(checkpoint, window_idx)
        pred_norm = pred_norm[:horizon]
        start = window_idx + SEQ_LEN
        truth_norm = self.data.test_norm[start: start + horizon]

        mse, mae = _metrics(pred_norm, truth_norm)
        per_channel = {
            ch: dict(zip(("mse", "mae"), _metrics(pred_norm[:, i], truth_norm[:, i])))
            for i, ch in enumerate(self.channel_names)
        }
        dates = self.data.test_dates
        return {
            "window_idx": window_idx,
            "horizon": horizon,
            "checkpoint": checkpoint,
            "source": source,
            "channels": self.channel_names,
            "history_dates": dates[start - context_hours: start].astype(str).tolist(),
            "history": self.data.test_raw[start - context_hours: start],
            "dates": dates[start: start + horizon].astype(str).tolist(),
            "predictions": self.data.denormalize(pred_norm),
            "ground_truth": self.data.test_raw[start: start + horizon],
            "metrics": {"mse": mse, "mae": mae},
            "per_channel_metrics": per_channel,
        }

    def horizon_metrics(self, window_idx: int, checkpoint: str = "instnorm",
                        horizons: Sequence[int] = HORIZONS) -> Dict[int, Dict[str, float]]:
        pred_norm, _ = self._predict(checkpoint, window_idx)
        start = window_idx + SEQ_LEN
        out = {}
        for h in horizons:
            mse, mae = _metrics(pred_norm[:h], self.data.test_norm[start: start + h])
            out[h] = {"mse": mse, "mae": mae}
        return out

    def scenario_forecast(self, window_idx: int, horizon: int, checkpoint: str,
                          perturbations: Dict[str, float]) -> Dict:
        """
        Re-run the forecast after scaling the last 96 hours of the chosen
        channels by (1 + p) in real units. Both the base and the scenario are
        run live through the same pipeline, so the difference is the model's
        response to the change alone.
        """
        start = window_idx + SEQ_LEN
        raw = self.data.test_raw[window_idx: start].copy()
        for ch, p in perturbations.items():
            raw[-PATCH_LEN:, self.channel_names.index(ch)] *= (1 + p)
        scen_norm = self._live_rollout(checkpoint, self.data.normalize(raw).astype("float32"))
        base_norm = self._live_predict(checkpoint, window_idx)
        return {
            "channels": self.channel_names,
            "dates": self.data.test_dates[start: start + horizon].astype(str).tolist(),
            "base": self.data.denormalize(base_norm[:horizon]),
            "predictions": self.data.denormalize(scen_norm[:horizon]),
            "perturbations": perturbations,
        }

    def backtest(self, checkpoint: str = "instnorm", num_windows: int = 50,
                 horizons: Sequence[int] = HORIZONS) -> Dict:
        """
        Walk-forward backtest over `num_windows` evenly spaced test windows
        (drawn from the precomputed set when there is one, so it's instant).
        """
        candidates = sorted(self.precomputed.get(checkpoint, {})) or self.windows(checkpoint)
        if not candidates:
            raise RuntimeError(f"No forecasts available for '{checkpoint}'")
        num_windows = min(num_windows, len(candidates))
        picks = np.linspace(0, len(candidates) - 1, num_windows).round().astype(int)
        windows = [candidates[i] for i in sorted(set(picks.tolist()))]

        per_h = {h: {"mse": [], "mae": []} for h in horizons}
        for w in windows:
            for h, m in self.horizon_metrics(w, checkpoint, horizons).items():
                per_h[h]["mse"].append(m["mse"])
                per_h[h]["mae"].append(m["mae"])

        results = {
            h: {
                "mse_mean": float(np.mean(v["mse"])),
                "mse_std": float(np.std(v["mse"])),
                "mae_mean": float(np.mean(v["mae"])),
                "mae_std": float(np.std(v["mae"])),
                "num_windows": len(v["mse"]),
            }
            for h, v in per_h.items()
        }
        return {"results": results, "windows": windows}
