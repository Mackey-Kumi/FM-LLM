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
