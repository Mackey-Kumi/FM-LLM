"""
Turning a trained model into forecasts of any length (autoregressive rollout,
as in experiments 06-10), plus the simple rules every forecast is compared with.
"""

import numpy as np
import torch

from .model import FMLLM, instance_denormalize, instance_normalize


def autoregressive_rollout(initial_context, predict_fn, num_steps):
    """
    Forecast further than one patch by feeding predictions back in. Each step
    predicts the next patch, then slides the context window: drop the oldest
    token, append the prediction (paper Section 3.3: "the recursive use of model
    outputs as future inputs").

    Args:
        initial_context: (batch, channels, num_tokens, patch_len), the real history as patches.
        predict_fn: function mapping a context of that shape to the next patch
            (batch, channels, patch_len).
        num_steps: how many patches to generate.

    Returns:
        (batch, channels, num_steps, patch_len), the generated patches in order.
    """
    context = initial_context
    generated = []
    for _ in range(num_steps):
        next_patch = predict_fn(context)
        generated.append(next_patch)
        context = torch.cat([context[:, :, 1:, :], next_patch.unsqueeze(2)], dim=2)
    return torch.stack(generated, dim=2)


@torch.no_grad()
def forecast(model: FMLLM, history, horizon):
    """
    Forecast any number of hours ahead. Horizons up to patch_len (e.g. our 48 h,
    with patch 96) are a single model step. Longer ones use autoregressive_rollout.
    The instance-norm statistics come from the real history only and stay fixed
    for the whole rollout.

    Args:
        model: a trained FMLLM (put it in eval mode first).
        history: (batch, channels, seq_len) tensor, on the scale the model was trained on.
        horizon: hours to forecast.

    Returns:
        (batch, channels, horizon) on the same scale as `history`.
    """
    c = model.config
    if c.instance_norm:
        history, mean, std = instance_normalize(history)
    tokens = history.unfold(-1, c.patch_len, c.patch_len)
    steps = -(-horizon // c.patch_len)
    generated = autoregressive_rollout(tokens, lambda ctx: model.predict_tokens(ctx)[0].float(), steps)
    b, ch, n, p = generated.shape
    pred = generated.reshape(b, ch, n * p)[:, :, :horizon]
    if c.instance_norm:
        pred = instance_denormalize(pred, mean, std)
    return pred


BASELINES = {
    "persistence": "Same as the last hour",
    "daily_naive": "Same as the last 24 hours",
    "weekly_naive": "Same as the same hours last week",
}


def baseline_forecast(history: np.ndarray, method: str, horizon: int) -> np.ndarray:
    """
    The simple rules every model forecast is compared against (see BASELINES).

    Args:
        history: (seq_len, channels) past values, in any units.
        method: "persistence" (repeat the last hour), "daily_naive" (repeat the
            last 24 h) or "weekly_naive" (repeat the last 168 h).
        horizon: hours to forecast.

    Returns:
        (horizon, channels), in the same units as `history`.
    """
    if method == "persistence":
        return np.repeat(history[-1:], horizon, axis=0)
    period = {"daily_naive": 24, "weekly_naive": 168}[method]
    reps = -(-horizon // period)
    return np.tile(history[-period:], (reps, 1))[:horizon]


# ---------------------------------------------------------------------------
# Uncertainty: Monte Carlo dropout ensemble
# ---------------------------------------------------------------------------
def enable_dropout(model: FMLLM):
    """
    Put the model in evaluation mode, then switch ONLY its dropout layers back on.
    Each forward pass then randomly drops a different 20% of some internal values,
    so repeated forecasts differ slightly. That spread is our uncertainty ("Monte
    Carlo dropout").

    Args:
        model: a trained FMLLM.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def forecast_ensemble(model: FMLLM, history, horizon, n_samples: int = 30, chunk: int = 256):
    """
    Forecast the same histories `n_samples` times with dropout on.

    Args:
        model: a trained FMLLM. It's left in eval mode afterwards (dropout off again).
        history: (batch, channels, seq_len) tensor, on the scale the model was trained on.
        horizon: hours to forecast.
        n_samples: ensemble size (30 is a common choice: enough for stable percentiles).
        chunk: how many histories go through the model at once (lower it if memory runs out).

    Returns:
        (n_samples, batch, channels, horizon) tensor on the same scale as `history`.
    """
    enable_dropout(model)
    try:
        repeated = history.repeat(n_samples, 1, 1)  # sample-major: rows [k*batch:(k+1)*batch] = sample k
        outs = [forecast(model, repeated[i: i + chunk], horizon) for i in range(0, len(repeated), chunk)]
        return torch.cat(outs).reshape(n_samples, *history.shape[:2], horizon)
    finally:
        model.eval()


def summarize_ensemble(samples: np.ndarray, limit: float, hours: int = 24,
                       quantiles=(0.1, 0.5, 0.9)) -> dict:
    """
    Turn an ensemble into what the app shows: a central forecast, a range, and
    the likelihood of a bad day.

    Args:
        samples: (n_samples, horizon) ensemble for ONE sensor, in µg/m³.
        limit: level for the likelihood, e.g. 35 (Ghana's 24-hour standard).
        hours: the likelihood is P(average over the first `hours` hours >= limit).
        quantiles: which percentiles to return for every hour.

    Returns:
        dict with
            median, low, high: (horizon,) arrays (the 50th, first and last of `quantiles`),
            prob_exceed: share of ensemble members whose first-`hours` average reaches `limit`,
            mean_first_hours: (n_samples,) each member's first-`hours` average (for inspection).
    """
    q = np.quantile(samples, quantiles, axis=0)
    means = samples[:, :hours].mean(axis=1)
    return {"low": q[0], "median": q[len(quantiles) // 2], "high": q[-1],
            "prob_exceed": float((means >= limit).mean()), "mean_first_hours": means}
