"""
FM-LLM architecture: Fourier Embedding Module (paper Eq. 7-11) -> frozen
Llama-3.2-1B -> FAN-MoE decoder (Eq. 13-18), plus our per-window instance
normalization (experiment 09).

Data flow for one forward pass (B = batch, C = channels/sensors, N = tokens, P = patch_len):

    history (B, C, seq_len)
      -> instance_normalize                     each window rescaled by its own mean/std
      -> unfold into patches   (B, C, N, P)     N = seq_len / P tokens, e.g. 672 / 96 = 7
      -> FourierEmbeddingModule (B, C, N, 2048) each patch -> one Llama-sized vector
      -> Llama-3.2-1B, frozen  (B, C, N, 2048)  reads the token sequence
      -> FANMoEDecoder         (B, C, N, P)     each output token -> a patch of P hours
      -> keep the last token   (B, C, P)        = the forecast for the next P hours
      -> instance_denormalize                   back to the input's scale

The modules and their attribute names are identical to
kaggle/experiments/09_instance_norm_retrain, so the ETTh1 checkpoints load
into them unchanged (tests/test_fmllm.py checks the outputs match exactly).
What's new here is `ModelConfig` (patch and history lengths are no longer
hard-coded) and a swappable backbone, so tests and CPU smoke runs can use a
tiny stand-in instead of downloading Llama.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    """
    All the sizes that define an FM-LLM model. The defaults are our ETTh1 settings (paper Table 2).

    Args:
        patch_len: hours per patch; one patch = one token for the language model.
            It's also the forecast length of one step.
        seq_len: hours of history the model reads. Must be a multiple of patch_len
            (672 = 28 days = 7 tokens).
        mlp_dim: width of the Fourier Embedding Module's hidden layer.
        llm_dim: hidden size of the backbone (2048 for Llama-3.2-1B).
        hidden_dim: width of each expert's hidden layer in the decoder.
        num_shared: number of Fourier experts, which run on every token.
        num_routed: number of plain feed-forward experts, chosen per token by the router.
        top_k: how many routed experts each token uses.
        dropout: dropout rate in the trainable modules (paper Table 2: 0.2 for ETTh1).
        p_ratio: share of each FAN layer's width given to the trend (non-periodic) part.
            Our own choice; the paper doesn't state it.
        instance_norm: rescale every input window by its own mean/std (our experiment 09 addition).
    """
    patch_len: int = 96
    seq_len: int = 672
    mlp_dim: int = 512
    llm_dim: int = 2048
    hidden_dim: int = 512
    num_shared: int = 2
    num_routed: int = 2
    top_k: int = 1
    dropout: float = 0.2
    p_ratio: float = 0.25
    instance_norm: bool = True

    @property
    def num_tokens(self) -> int:
        """Returns: tokens per history window (seq_len / patch_len)."""
        return self.seq_len // self.patch_len

    def __post_init__(self):
        """Raises: ValueError if seq_len isn't a whole number of patches."""
        if self.seq_len % self.patch_len:
            raise ValueError(f"seq_len ({self.seq_len}) must be a multiple of patch_len ({self.patch_len})")


# ---------------------------------------------------------------------------
# Fourier Embedding Module (Eq. 7-11)
# ---------------------------------------------------------------------------
class FourierAnalysisNetwork(nn.Module):
    """
    One FAN layer (paper Eq. 7-8): splits its output into periodic features,
    [cos(W_p x), sin(W_p x)], and trend features, SiLU(W_p̄ x + b).

    Args:
        hidden_dim: input and output width (the output keeps the same width as the input).
        p_ratio: share of the output width given to the trend part. The rest is split
            equally between cos and sin.
    """

    def __init__(self, hidden_dim=512, p_ratio=0.25):
        super().__init__()
        d_pbar = max(1, round(hidden_dim * p_ratio))
        d_p = (hidden_dim - d_pbar) // 2
        d_pbar = hidden_dim - 2 * d_p          # so cos + sin + trend add up to hidden_dim exactly
        self.w_p = nn.Linear(hidden_dim, d_p, bias=False)
        self.w_pbar = nn.Linear(hidden_dim, d_pbar)

    def forward(self, x):
        """
        Args:
            x: (..., hidden_dim)

        Returns:
            (..., hidden_dim): [cos | sin | trend] concatenated on the last dimension.
        """
        periodic = self.w_p(x)
        trend = F.silu(self.w_pbar(x))
        return torch.cat([torch.cos(periodic), torch.sin(periodic), trend], dim=-1)


class FourierEmbeddingModule(nn.Module):
    """
    Turns each patch of raw values into one vector the language model can read
    (paper Eq. 9-11): Linear -> SiLU -> Dropout -> FAN -> Tanh -> Linear.

    Args:
        patch_len: hours per patch (input width).
        mlp_dim: hidden width.
        llm_dim: output width, the backbone's hidden size.
        dropout: dropout rate after the first activation.
        p_ratio: FAN trend share (see FourierAnalysisNetwork).
    """

    def __init__(self, patch_len, mlp_dim, llm_dim, dropout=0.2, p_ratio=0.25):
        super().__init__()
        self.linear_in = nn.Linear(patch_len, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.fan = FourierAnalysisNetwork(hidden_dim=mlp_dim, p_ratio=p_ratio)
        self.linear_out = nn.Linear(mlp_dim, llm_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, tokens, patch_len), normalized patch values.

        Returns:
            (batch, channels, tokens, llm_dim) token embeddings.
        """
        x = self.linear_in(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fan(x)
        x = torch.tanh(x)
        return self.linear_out(x)


# ---------------------------------------------------------------------------
# FAN-MoE decoder (Eq. 13-18)
# ---------------------------------------------------------------------------
class FANExpert(nn.Module):
    """
    A shared "Fourier expert" (paper Eq. 13-14), always on, specialised in
    repeating patterns: Linear -> Tanh -> Dropout -> FAN -> SiLU -> Dropout -> Linear.

    Args:
        llm_dim: input width (the backbone's hidden size).
        hidden_dim: hidden width.
        patch_len: output width, the hours predicted per token.
        dropout: dropout rate.
        p_ratio: FAN trend share.
    """

    def __init__(self, llm_dim, hidden_dim, patch_len, dropout=0.2, p_ratio=0.25):
        super().__init__()
        self.linear_in = nn.Linear(llm_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fan = FourierAnalysisNetwork(hidden_dim=hidden_dim, p_ratio=p_ratio)
        self.dropout2 = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        """
        Args:
            x: (..., llm_dim) backbone output.

        Returns:
            (..., patch_len) this expert's forecast for each token.
        """
        x = torch.tanh(self.linear_in(x))
        x = self.dropout1(x)
        x = self.fan(x)
        x = F.silu(x)
        x = self.dropout2(x)
        return self.linear_out(x)


class RoutedExpert(nn.Module):
    """
    A routed expert (paper Eq. 15): a plain two-layer feed-forward network for
    irregular patterns, used only when the router picks it.

    Args:
        llm_dim: input width.
        hidden_dim: hidden width.
        patch_len: output width.
    """

    def __init__(self, llm_dim, hidden_dim, patch_len):
        super().__init__()
        self.layer1 = nn.Linear(llm_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, patch_len)

    def forward(self, x):
        """
        Args:
            x: (..., llm_dim)

        Returns:
            (..., patch_len)
        """
        return self.layer2(F.silu(self.layer1(x)))


class MoEGate(nn.Module):
    """
    The router (paper Eq. 16-17): scores every routed expert for each token
    and keeps the top-k. A per-expert bias, adjusted by update_bias rather than
    by gradients, steers it away from always choosing the same expert.

    Args:
        llm_dim: input width.
        num_routed: number of routed experts to choose from.
        top_k: experts chosen per token.
        bias_update_rate: how far update_bias moves each expert's bias per step.
    """

    def __init__(self, llm_dim, num_routed, top_k, bias_update_rate=1e-3):
        super().__init__()
        self.top_k = top_k
        self.num_routed = num_routed
        self.bias_update_rate = bias_update_rate
        self.score_proj = nn.Linear(llm_dim, num_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_routed))

    def forward(self, x):
        """
        Args:
            x: (..., llm_dim)

        Returns:
            gate_weights: (..., top_k) mixing weights of the chosen experts (they sum to 1).
            top_indices: (..., top_k) which experts were chosen.
            scores: (..., num_routed) softmax scores for every expert (used by the balance loss).
        """
        scores = F.softmax(self.score_proj(x), dim=-1)
        biased_scores = scores + self.expert_bias          # the bias affects the choice...
        _, top_indices = torch.topk(biased_scores, self.top_k, dim=-1)
        selected_raw = torch.gather(scores, -1, top_indices)  # ...but not the mixing weights
        gate_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)
        return gate_weights, top_indices, scores

    @torch.no_grad()
    def update_bias(self, top_indices):
        """
        Load balancing without an extra loss term: experts chosen less often
        than average get their bias raised, over-used ones get it lowered.
        Called once per training step.

        Args:
            top_indices: (..., top_k) the experts chosen in this batch, from forward().
        """
        load = torch.zeros(self.num_routed, device=top_indices.device)
        flat = top_indices.reshape(-1)
        load.scatter_add_(0, flat, torch.ones_like(flat, dtype=load.dtype))
        target = load.mean()
        self.expert_bias += self.bias_update_rate * torch.sign(target - load)


class FANMoEDecoder(nn.Module):
    """
    Turns the backbone's output back into numbers (paper Eq. 18): the sum of all
    Fourier experts plus the gate-weighted routed experts.

    Args:
        llm_dim: input width.
        hidden_dim: hidden width of each expert.
        patch_len: hours predicted per token.
        num_shared: number of Fourier experts (always on).
        num_routed: number of routed experts.
        top_k: routed experts per token.
        dropout: dropout rate inside the Fourier experts.
        p_ratio: FAN trend share.
    """

    def __init__(self, llm_dim, hidden_dim, patch_len, num_shared, num_routed, top_k, dropout=0.2, p_ratio=0.25):
        super().__init__()
        self.shared_experts = nn.ModuleList(
            [FANExpert(llm_dim, hidden_dim, patch_len, dropout=dropout, p_ratio=p_ratio) for _ in range(num_shared)]
        )
        self.routed_experts = nn.ModuleList(
            [RoutedExpert(llm_dim, hidden_dim, patch_len) for _ in range(num_routed)]
        )
        self.gate = MoEGate(llm_dim, num_routed, top_k)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, tokens, llm_dim) backbone output.

        Returns:
            forecast: (batch, channels, tokens, patch_len), one patch per token.
            top_indices: (batch, channels, tokens, top_k) routing choices.
            scores: (batch, channels, tokens, num_routed) router scores.
        """
        shared_out = sum(expert(x) for expert in self.shared_experts)
        gate_weights, top_indices, scores = self.gate(x)
        # weight 0 for the experts that weren't chosen, so all experts can be summed the same way
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
# Frozen Llama-3.2-1B backbone
# ---------------------------------------------------------------------------
HF_TOKEN_FILES = (
    "/kaggle/input/fm-llm-hf-token/hf_token.txt",
    "/kaggle/input/datasets/mackeykumi/fm-llm-hf-token/hf_token.txt",
)


def hf_token() -> Optional[str]:
    """
    Find the Hugging Face token needed to download Llama (a gated model).

    Returns:
        HF_TOKEN from the environment or .env, else the token from the private Kaggle
        dataset, else None.
    """
    try:
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel  # optional, local runs only
        load_dotenv()
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for path in HF_TOKEN_FILES:
        if os.path.exists(path):
            return Path(path).read_text(encoding="utf-8").strip()
    return None


class LlamaBackbone(nn.Module):
    """
    Llama-3.2-1B with every weight frozen (never trained). It runs in bfloat16 on GPU,
    as in training, and float32 on CPU.

    Args:
        device: where to load it, e.g. "cpu" or torch.device("cuda:0").
    """

    def __init__(self, device="cpu"):
        super().__init__()
        # pylint: disable-next=import-outside-toplevel  # heavy; tests and CPU smoke runs never load it
        from transformers import LlamaModel

        self.dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.llama = LlamaModel.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=self.dtype, low_cpu_mem_usage=True, token=hf_token(),
        )
        for param in self.llama.parameters():
            param.requires_grad = False
        self.llama.to(device).eval()

    def forward(self, embedding_inputs):
        """
        Each channel's token sequence goes through Llama separately (channel independence).

        Args:
            embedding_inputs: (batch, channels, tokens, 2048) from the embedding module.

        Returns:
            (batch, channels, tokens, 2048) Llama's last hidden layer.
        """
        b, s, n, h = embedding_inputs.shape
        flat_inputs = embedding_inputs.reshape(b * s, n, h).to(self.dtype)  # every channel = one sequence
        outputs = self.llama(inputs_embeds=flat_inputs)
        return outputs.last_hidden_state.reshape(b, s, n, h)


class TinyBackbone(nn.Module):
    """
    CPU stand-in for Llama in smoke runs and tests. It has the same input and output
    shape, and a running mean over tokens so later tokens can see earlier ones, like
    Llama's causal attention. Never use it for real results.

    Args:
        dim: hidden size (use the config's llm_dim).
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, tokens, dim)

        Returns:
            (batch, channels, tokens, dim)
        """
        steps = torch.arange(1, x.shape[2] + 1, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        return torch.tanh(self.proj(x.cumsum(dim=2) / steps))


# ---------------------------------------------------------------------------
# Instance (per-window) normalization -- experiment 09
# ---------------------------------------------------------------------------
def instance_normalize(history):
    """
    Rescale each window to mean 0 and spread 1, using that window's own statistics.
    The model then sees the shape of the signal, not its level. That's why one model
    can serve a dusty market and a quiet street.

    Args:
        history: (batch, channels, seq_len)

    Returns:
        normalized: (batch, channels, seq_len)
        mean: (batch, channels, 1), each window's mean (needed to undo the scaling).
        std: (batch, channels, 1), each window's standard deviation (+1e-5 so it's never 0).
    """
    mean = history.mean(dim=-1, keepdim=True)
    std = history.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
    return (history - mean) / std, mean, std


def instance_denormalize(x, mean, std):
    """
    Undo instance_normalize on a prediction.

    Args:
        x: (batch, channels, length), a prediction on the normalized scale.
        mean, std: (batch, channels, 1) from instance_normalize of the same history.

    Returns:
        (batch, channels, length) on the history's original scale.
    """
    return x * std + mean


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class FMLLM(nn.Module):
    """
    The full model: history -> forecast of the next patch_len hours (see the module
    docstring for the data flow). Every channel is forecast from its own history only
    (channel independence). Only the embedding module and the decoder are trained;
    the backbone stays frozen.

    Args:
        config: model sizes (ModelConfig).
        backbone: LlamaBackbone for real runs, or TinyBackbone(config.llm_dim) for tests.
    """

    def __init__(self, config: ModelConfig, backbone: nn.Module):
        super().__init__()
        self.config = config
        c = config
        self.embedding_module = FourierEmbeddingModule(c.patch_len, c.mlp_dim, c.llm_dim, c.dropout, c.p_ratio)
        self.llama_backbone = backbone
        self.decoder = FANMoEDecoder(c.llm_dim, c.hidden_dim, c.patch_len, c.num_shared, c.num_routed,
                                     c.top_k, c.dropout, c.p_ratio)
        # A plain attribute, not read from the decoder's parameters: nn.DataParallel's
        # per-GPU copies have no registered parameters, so looking one up there fails.
        self.decoder_dtype = torch.float32

    def set_decoder_dtype(self, dtype):
        """
        Cast the decoder to a number format (we use bfloat16 on GPU, as in the ETTh1
        experiments) and remember it, so the backbone output is cast to match.

        Args:
            dtype: e.g. torch.bfloat16 or torch.float32.

        Returns:
            self, for chaining.
        """
        self.decoder.to(dtype)
        self.decoder_dtype = dtype
        return self

    def trainable_parameters(self):
        """Returns: list of the parameters the optimizer updates (embedding module + decoder, not the backbone)."""
        return list(self.embedding_module.parameters()) + list(self.decoder.parameters())

    def predict_tokens(self, tokens):
        """
        One pass through embedding -> backbone -> decoder, without normalization.
        Used directly by the rollout in forecast.py.

        Args:
            tokens: (batch, channels, num_tokens, patch_len), already normalized.

        Returns:
            next_patch: (batch, channels, patch_len), the decoder output at the LAST token,
                i.e. the prediction for the hours right after the history.
            top_indices, scores: routing information (see FANMoEDecoder.forward).
        """
        embeds = self.embedding_module(tokens)
        llm_out = self.llama_backbone(embeds)
        forecast, top_indices, scores = self.decoder(llm_out.to(self.decoder_dtype))
        return forecast[:, :, -1, :], top_indices, scores

    def forward(self, history):
        """
        Args:
            history: (batch, channels, seq_len) on whatever scale the data is in.

        Returns:
            pred: (batch, channels, patch_len), the forecast on the same scale as `history`.
            top_indices, scores: routing information (needed for the balance loss in training).
        """
        if self.config.instance_norm:
            history, mean, std = instance_normalize(history)
        tokens = history.unfold(-1, self.config.patch_len, self.config.patch_len)  # (B, C, N, P)
        pred, top_indices, scores = self.predict_tokens(tokens)
        if self.config.instance_norm:
            pred = instance_denormalize(pred.float(), mean, std)
        return pred, top_indices, scores


def save_checkpoint(model: FMLLM, path, **extra):
    """
    Save the trained parts only. The frozen backbone is never saved, since it's always
    re-downloaded.

    Args:
        model: the FMLLM to save.
        path: file to write (.pt).
        **extra: anything else to store alongside, e.g. epoch=3, val_loss=0.31.
    """
    torch.save({
        "embedding_module": model.embedding_module.state_dict(),
        "decoder": model.decoder.state_dict(),
        "config": asdict(model.config),
        **extra,
    }, path)


def load_checkpoint(path, backbone: nn.Module, device="cpu", config: Optional[ModelConfig] = None) -> FMLLM:
    """
    Rebuild a model from a checkpoint. Works for our new checkpoints, which store their
    config, and for the ETTh1 ones from experiments 03-09, which don't: there, pass
    `config`, or the defaults (patch 96, history 672) are used.

    Args:
        path: checkpoint file.
        backbone: the backbone to plug in (LlamaBackbone or TinyBackbone).
        device: where to put the trainable parts.
        config: overrides the stored config. Needed only for checkpoints that don't
            store one and don't use the defaults.

    Returns:
        FMLLM in eval mode (dropout off).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = config or ModelConfig(**ckpt.get("config", {}))
    model = FMLLM(config, backbone)
    model.embedding_module.load_state_dict(ckpt["embedding_module"])
    model.decoder.load_state_dict(ckpt["decoder"])
    model.embedding_module.to(device)
    model.decoder.to(device)
    return model.eval()
