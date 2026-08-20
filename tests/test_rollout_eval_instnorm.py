import importlib.util
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "kaggle"
    / "experiments"
    / "10_rollout_eval_instnorm"
    / "kaggle_eval_rollout.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("rollout_eval_instnorm_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load_module()


def test_autoregressive_rollout_slides_window_and_orders_output(m):
    batch, channels, num_tokens, patch_len = 1, 1, 3, 2
    initial_context = torch.zeros(batch, channels, num_tokens, patch_len)

    call_log = []

    def fake_predict(context):
        call_log.append(context.clone())
        step = len(call_log)
        return torch.full((batch, channels, patch_len), float(step))

    generated = m.autoregressive_rollout(initial_context, fake_predict, num_steps=5)

    assert generated.shape == (batch, channels, 5, patch_len)
    for step in range(5):
        assert torch.all(generated[:, :, step, :] == float(step + 1))


def test_truncate_to_horizon_slices_flattened_sequence(m):
    batch, channels, num_steps, patch_len = 1, 1, 8, 96
    generated = torch.arange(num_steps * patch_len, dtype=torch.float32).reshape(
        batch, channels, num_steps, patch_len
    )
    for horizon in (192, 336, 720):
        truncated = m.truncate_to_horizon(generated, horizon)
        assert truncated.shape == (batch, channels, horizon)
        assert torch.equal(truncated[0, 0], torch.arange(horizon, dtype=torch.float32))


def test_forecast_window_dataset_length_matches_known_test_set_size(m):
    dummy_data = np.zeros((672 + 2880, 7), dtype="float32")
    dataset = m.ForecastWindowDataset(dummy_data, seq_len=672, future_len=96)
    assert len(dataset) == 2785


def test_full_pipeline_shape_with_dummy_backbone(m):
    class _IdentityBackbone(nn.Module):
        def forward(self, embedding_inputs):
            return embedding_inputs

    torch.manual_seed(0)
    pred_len = 96
    embedding_module = m.FourierEmbeddingModule(patch_len=pred_len, mlp_dim=32, llm_dim=64)
    decoder = m.FANMoEDecoder(
        llm_dim=64, hidden_dim=32, patch_len=pred_len,
        num_shared=2, num_routed=2, top_k=1,
    )
    backbone = _IdentityBackbone()
    embedding_module.eval()
    decoder.eval()

    batch, channels, num_tokens = 2, 4, 7
    tokens = torch.randn(batch, channels, num_tokens, pred_len)
    pred_patch = m.predict_next_patch(tokens, embedding_module, backbone, decoder)

    assert pred_patch.shape == (batch, channels, pred_len)


def test_truncate_then_denormalize_recovers_original_scale(m):
    """The real eval loop truncates a rollout trajectory to a horizon
    (batch, channels, horizon) and only then denormalizes it against the
    fixed per-window mean/std -- verify that composition broadcasts
    correctly and actually recovers the original scale."""
    torch.manual_seed(0)
    history = torch.randn(2, 3, 672) * 5 + 10
    _, mean, std = m.instance_normalize(history)

    batch, channels, num_steps, patch_len = 2, 3, 8, 96
    generated_norm = torch.randn(batch, channels, num_steps, patch_len)

    for horizon in (192, 336, 720):
        truncated_norm = m.truncate_to_horizon(generated_norm, horizon)
        truncated_actual = m.instance_denormalize(truncated_norm, mean, std)
        assert truncated_actual.shape == (batch, channels, horizon)
        expected = truncated_norm * std + mean
        assert torch.allclose(truncated_actual, expected)
