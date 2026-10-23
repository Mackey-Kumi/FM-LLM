"""Hybrid time-frequency loss + sequence balance loss (paper Eq. 19-23), as in experiments 03-09."""

import torch
import torch.nn.functional as F


def hybrid_loss(pred, target, alpha=1.0, beta=1.0):
    """
    The training loss (paper Eq. 19-21): a time-domain error plus a frequency-domain error.

    Both terms weight hour l by 1/sqrt(l) ("signal decay"), so mistakes in the
    first hours of the forecast cost more than mistakes far ahead. The
    frequency term compares the Fourier transforms of prediction and target, so
    the model is also penalised for getting the rhythm wrong (e.g. the daily
    cycle), not just the values.

    Args:
        pred: (..., L) forecast, any leading dimensions, L = hours forecast.
        target: same shape, what actually happened.
        alpha: weight of the frequency term.
        beta: weight of the time term.

    Returns:
        Scalar tensor: alpha * frequency_error + beta * time_error.
    """
    # L, T, P below are the symbols of the paper's Eq. 19-23, kept so the code reads against the paper.
    # pylint: disable=invalid-name
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
    """
    A small penalty (paper Eq. 22-23) that grows when the router keeps sending
    tokens to the same routed expert, so every expert gets used and trained.

    Args:
        scores: (..., num_routed) router softmax scores, from MoEGate.
        top_indices: (..., top_k) experts chosen, from MoEGate.
        num_routed: number of routed experts.
        top_k: experts chosen per token.
        gamma: strength of the penalty.

    Returns:
        Scalar tensor: gamma * sum over experts of (how often it was chosen x its average score).
    """
    # T, P are the symbols of the paper's Eq. 22-23.
    # pylint: disable=invalid-name
    T = top_indices.shape[:-1].numel()
    one_hot = F.one_hot(top_indices, num_classes=num_routed).sum(dim=-2)
    f = one_hot.reshape(-1, num_routed).sum(dim=0).float() * (num_routed / (top_k * T))
    s_norm = scores / scores.sum(dim=-1, keepdim=True)
    P = s_norm.reshape(-1, num_routed).mean(dim=0)
    return gamma * (f * P).sum()
