"""
GGUF-backed Inference Engine for GridForecast
Uses transformers for backbone (with pre-caching) for correct hidden states.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import pickle

# Model constants (matching training)
DROPOUT = 0.2
PATCH_LEN = 96
SEQ_LEN = 672
LLM_DIM = 2048
MLP_DIM = 512
HIDDEN_DIM = 512
NUM_SHARED = 2
NUM_ROUTED = 2
TOP_K = 1

MAX_HORIZON = 720
NUM_ROLLOUT_STEPS = -(-MAX_HORIZON // PATCH_LEN)  # 8

CHECKPOINT_DIR = Path(__file__).parent.parent / "kaggle" / "datasets" / "checkpoint"
CHECKPOINT_SPECS = {
    "instnorm": ("etth1_96_instnorm_best.pt", True),  # Best model
    "dropout_longer": ("etth1_96_dropout_longer_best.pt", False),
    "dropout": ("etth1_96_dropout_best.pt", False),
    "no_dropout": ("etth1_96_best.pt", False),
}

HORIZONS = (96, 192, 336, 720)

# Global normalization stats (will be computed from data)
CHANNEL_MEAN = None
CHANNEL_STD = None
CHANNEL_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# Cache directory for pre-computed backbone outputs
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
BACKBONE_CACHE_FILE = CACHE_DIR / "backbone_outputs.pkl"


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


class LlamaBackboneCPU(nn.Module):
    """CPU-compatible Llama backbone using transformers (for pre-caching)"""
    def __init__(self, device="cpu"):
        super().__init__()
        print("Loading Llama-3.2-1B backbone (CPU)...")
        from transformers import LlamaModel
        import os
        hf_token = os.environ.get("HF_TOKEN")
        self.llama = LlamaModel.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        for param in self.llama.parameters():
            param.requires_grad = False
        self.llama.to(device)
        self.llama.eval()
        print("Llama-3.2-1B loaded on", device)

    def forward(self, embedding_inputs):
        b, s, n, h = embedding_inputs.shape
        flat_inputs = embedding_inputs.reshape(b * s, n, h).to(torch.float32)
        with torch.no_grad():
            outputs = self.llama(inputs_embeds=flat_inputs)
        return outputs.last_hidden_state.reshape(b, s, n, h)


class CachedBackbone(nn.Module):
    """
    Uses pre-computed backbone outputs for instant inference.
    Run `python -m app.precache_backbone` once to generate cache.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.cache = self._load_cache()
        print(f"Loaded backbone cache: {len(self.cache)} entries")

    def _load_cache(self):
        if BACKBONE_CACHE_FILE.exists():
            with open(BACKBONE_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        else:
            print("WARNING: No backbone cache found. Run `python -m app.precache_backbone` first.")
            return {}

    def forward(self, embedding_inputs):
        # Create cache key from input tensor (rounded for matching)
        key = tuple(embedding_inputs.cpu().numpy().round(6).tobytes())
        if key in self.cache:
            return self.cache[key]
        
        # Fallback: compute on the fly (slow)
        print("Cache miss - computing backbone on the fly...")
        if not hasattr(self, '_fallback_backbone'):
            self._fallback_backbone = LlamaBackboneCPU(self.device)
        with torch.no_grad():
            output = self._fallback_backbone(embedding_inputs)
        return output


def build_backbone_cache():
    """Pre-compute backbone outputs for all test windows"""
    print("Building backbone cache...")
    
    device = "cpu"
    backbone = LlamaBackboneCPU(device)
    
    # Load test data
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    raw_dataframe = pd.read_csv(url)
    numeric_data = raw_dataframe.drop(columns=["date"])
    
    HOURS_PER_MONTH = 30 * 24
    num_train = 12 * HOURS_PER_MONTH
    num_val = 4 * HOURS_PER_MONTH
    num_test = 4 * HOURS_PER_MONTH
    
    train_end = num_train
    val_end = num_train + num_val
    test_end = num_train + num_val + num_test
    
    train_data = numeric_data.iloc[0:train_end]
    test_data = numeric_data.iloc[val_end - SEQ_LEN : test_end]
    
    channel_mean = train_data.mean(axis=0).values.astype("float32")
    channel_std = train_data.std(axis=0).values.astype("float32")
    
    def normalize(df):
        return (df - channel_mean) / channel_std
    
    test_norm = normalize(test_data).values.astype("float32")
    
    # Load embedding module (needed to create inputs)
    embedding_module = FourierEmbeddingModule().to(device)
    embedding_module.eval()
    
    # Load a checkpoint to get embedding_module weights
    checkpoint_path = CHECKPOINT_DIR / "etth1_96_instnorm_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    embedding_module.load_state_dict(checkpoint["embedding_module"])
    
    max_windows = len(test_norm) - SEQ_LEN - MAX_HORIZON + 1
    cache = {}
    
    print(f"Processing {max_windows} test windows...")
    
    for i in range(max_windows):
        if i % 100 == 0:
            print(f"  Window {i}/{max_windows}")
        
        history = test_norm[i : i + SEQ_LEN]
        history_tensor = torch.from_numpy(history.T).unsqueeze(0).float().to(device)
        
        # Instance normalize (same as instnorm checkpoint)
        mean = history_tensor.mean(dim=-1, keepdim=True)
        std = history_tensor.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        history_input = (history_tensor - mean) / std
        
        # Tokenize
        history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)
        
        # Get embeddings
        with torch.no_grad():
            embeds = embedding_module(history_tokens)
        
        # Get backbone output
        with torch.no_grad():
            backbone_out = backbone(embeds)
        
        # Cache key
        key = tuple(embeds.cpu().numpy().round(6).tobytes())
        cache[key] = backbone_out.cpu()
    
    # Save cache
    with open(BACKBONE_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    
    print(f"✅ Cache saved to {BACKBONE_CACHE_FILE} ({len(cache)} entries)")


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


def load_etth1_data():
    """Load ETTh1 data and compute normalization stats"""
    global CHANNEL_MEAN, CHANNEL_STD
    
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    raw_dataframe = pd.read_csv(url)
    numeric_data = raw_dataframe.drop(columns=["date"])
    
    HOURS_PER_MONTH = 30 * 24
    num_train = 12 * HOURS_PER_MONTH
    num_val = 4 * HOURS_PER_MONTH
    num_test = 4 * HOURS_PER_MONTH
    
    train_end = num_train
    val_end = num_train + num_val
    test_end = num_train + num_val + num_test
    
    train_data = numeric_data.iloc[0:train_end]
    test_data = numeric_data.iloc[val_end - SEQ_LEN : test_end]
    
    CHANNEL_MEAN = train_data.mean(axis=0).values.astype("float32")
    CHANNEL_STD = train_data.std(axis=0).values.astype("float32")
    
    def normalize(df):
        return (df - CHANNEL_MEAN) / CHANNEL_STD
    
    test_norm = normalize(test_data).values.astype("float32")
    
    return test_norm, numeric_data.columns.tolist(), test_data


class ForecastEngine:
    """Main inference engine for the GridForecast application"""
    
    def __init__(self, checkpoint_name="instnorm", device="cpu"):
        self.device = device
        self.checkpoint_name = checkpoint_name
        
        print(f"Initializing ForecastEngine with {checkpoint_name}...")
        
        # Load model components
        self.embedding_module, self.decoder, self.use_instance_norm = load_checkpoint(
            checkpoint_name, device
        )
        
        # Use cached backbone
        self.backbone = CachedBackbone(device)
        
        # Load data and normalization stats
        self.test_norm, self.channel_names, self.test_raw = load_etth1_data()
        
        print(f"ForecastEngine ready. Test windows: {len(self.test_norm) - SEQ_LEN - MAX_HORIZON + 1}")
    
    def get_available_windows(self) -> int:
        return len(self.test_norm) - SEQ_LEN - MAX_HORIZON + 1
    
    def get_channel_names(self) -> List[str]:
        return self.channel_names
    
    def forecast_window(self, window_idx: int, horizon: int) -> Dict:
        """
        Run forecast for a specific test window.
        Returns predictions and ground truth for all channels.
        """
        # Extract history and future
        history = self.test_norm[window_idx : window_idx + SEQ_LEN]
        future = self.test_norm[window_idx + SEQ_LEN : window_idx + SEQ_LEN + horizon]
        
        # Prepare input: (1, channels, seq_len)
        history_tensor = torch.from_numpy(history.T).unsqueeze(0).float().to(self.device)
        
        if self.use_instance_norm:
            history_input, mean, std = instance_normalize(history_tensor)
        else:
            history_input = history_tensor
            mean = std = None
        
        # Tokenize: (1, channels, num_tokens, patch_len)
        history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)
        
        def predict_fn(context):
            return predict_next_patch(context, self.embedding_module, self.backbone, self.decoder)
        
        with torch.no_grad():
            generated = autoregressive_rollout(history_tokens, predict_fn, NUM_ROLLOUT_STEPS)
            pred_h = truncate_to_horizon(generated, horizon)
            
            if self.use_instance_norm:
                pred_h = instance_denormalize(pred_h, mean, std)
        
        # Denormalize to original scale
        pred_np = pred_h.squeeze(0).cpu().numpy().T  # (horizon, channels)
        pred_np = pred_np * CHANNEL_STD + CHANNEL_MEAN
        
        # Ground truth
        future_np = future  # (horizon, channels)
        
        return {
            "predictions": pred_np,
            "ground_truth": future_np,
            "channels": self.channel_names,
            "horizon": horizon,
            "window_idx": window_idx,
        }
    
    def forecast_all_horizons(self, window_idx: int) -> Dict[int, Dict]:
        """Run forecast for all horizons at once (single rollout)"""
        horizon = MAX_HORIZON
        
        history = self.test_norm[window_idx : window_idx + SEQ_LEN]
        future = self.test_norm[window_idx + SEQ_LEN : window_idx + SEQ_LEN + horizon]
        
        history_tensor = torch.from_numpy(history.T).unsqueeze(0).float().to(self.device)
        
        if self.use_instance_norm:
            history_input, mean, std = instance_normalize(history_tensor)
        else:
            history_input = history_tensor
            mean = std = None
        
        history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)
        
        def predict_fn(context):
            return predict_next_patch(context, self.embedding_module, self.backbone, self.decoder)
        
        with torch.no_grad():
            generated = autoregressive_rollout(history_tokens, predict_fn, NUM_ROLLOUT_STEPS)
            
            results = {}
            for h in HORIZONS:
                pred_h = truncate_to_horizon(generated, h)
                if self.use_instance_norm:
                    pred_h = instance_denormalize(pred_h, mean, std)
                
                pred_np = pred_h.squeeze(0).cpu().numpy().T
                pred_np = pred_np * CHANNEL_STD + CHANNEL_MEAN
                
                future_h = future[:h]
                
                mse = ((pred_np - future_h) ** 2).mean()
                mae = np.abs(pred_np - future_h).mean()
                
                results[h] = {
                    "predictions": pred_np,
                    "ground_truth": future_h,
                    "mse": mse,
                    "mae": mae,
                    "channels": self.channel_names,
                }
        
        return results
    
    def scenario_forecast(self, window_idx: int, horizon: int, 
                          channel_perturbations: Dict[str, float]) -> Dict:
        """
        Run forecast with modified input channels (scenario analysis).
        channel_perturbations: {channel_name: multiplier_or_delta}
        """
        history = self.test_norm[window_idx : window_idx + SEQ_LEN].copy()
        
        # Apply perturbations to the last window of history (most recent)
        for ch_name, perturbation in channel_perturbations.items():
            if ch_name in self.channel_names:
                ch_idx = self.channel_names.index(ch_name)
                # Apply to last 96 steps (one patch)
                history[-PATCH_LEN:, ch_idx] *= (1 + perturbation)
        
        history_tensor = torch.from_numpy(history.T).unsqueeze(0).float().to(self.device)
        
        if self.use_instance_norm:
            history_input, mean, std = instance_normalize(history_tensor)
        else:
            history_input = history_tensor
            mean = std = None
        
        history_tokens = history_input.unfold(-1, PATCH_LEN, PATCH_LEN)
        
        def predict_fn(context):
            return predict_next_patch(context, self.embedding_module, self.backbone, self.decoder)
        
        with torch.no_grad():
            generated = autoregressive_rollout(history_tokens, predict_fn, NUM_ROLLOUT_STEPS)
            pred_h = truncate_to_horizon(generated, horizon)
            
            if self.use_instance_norm:
                pred_h = instance_denormalize(pred_h, mean, std)
        
        pred_np = pred_h.squeeze(0).cpu().numpy().T
        pred_np = pred_np * CHANNEL_STD + CHANNEL_MEAN
        
        return {
            "predictions": pred_np,
            "channels": self.channel_names,
            "perturbations": channel_perturbations,
        }
    
    def backtest(self, num_windows: int = 50, horizons: Tuple[int, ...] = HORIZONS) -> Dict:
        """
        Walk-forward backtest on historical test windows.
        Returns aggregate metrics.
        """
        max_windows = self.get_available_windows()
        num_windows = min(num_windows, max_windows)
        
        results = {h: {"mse": [], "mae": []} for h in horizons}
        
        for i in range(num_windows):
            try:
                window_results = self.forecast_all_horizons(i)
                for h in horizons:
                    results[h]["mse"].append(window_results[h]["mse"])
                    results[h]["mae"].append(window_results[h]["mae"])
            except Exception as e:
                print(f"Window {i} failed: {e}")
                continue
        
        # Aggregate
        agg = {}
        for h in horizons:
            agg[h] = {
                "mse_mean": np.mean(results[h]["mse"]),
                "mse_std": np.std(results[h]["mse"]),
                "mae_mean": np.mean(results[h]["mae"]),
                "mae_std": np.std(results[h]["mae"]),
                "num_windows": len(results[h]["mse"]),
            }
        
        return agg


# Demo data preparation
def prepare_demo_data():
    """Download and save sample data for offline demo"""
    demo_dir = Path(__file__).parent / "demo_data"
    demo_dir.mkdir(exist_ok=True)
    
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    
    # Save full dataset
    df.to_csv(demo_dir / "etth1_full.csv", index=False)
    
    # Save last test window for quick demo
    HOURS_PER_MONTH = 30 * 24
    num_train = 12 * HOURS_PER_MONTH
    num_val = 4 * HOURS_PER_MONTH
    num_test = 4 * HOURS_PER_MONTH
    
    val_end = num_train + num_val
    test_end = num_train + num_val + num_test
    
    test_data = df.iloc[val_end - SEQ_LEN : test_end]
    test_data.to_csv(demo_dir / "etth1_test_window.csv", index=False)
    
    # Save normalization stats
    train_data = df.iloc[0:num_train].drop(columns=["date"])
    stats = {
        "mean": train_data.mean().to_dict(),
        "std": train_data.std().to_dict(),
    }
    with open(demo_dir / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Demo data saved to {demo_dir}")


if __name__ == "__main__":
    # Test the engine
    prepare_demo_data()
    print("Demo data prepared.")