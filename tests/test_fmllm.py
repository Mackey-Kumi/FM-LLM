import importlib.util
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from fmllm.events import assess_exceedance, score, score_by
from fmllm.forecast import autoregressive_rollout, baseline_forecast, forecast
from fmllm.losses import hybrid_loss, sequence_balance_loss
from fmllm.model import FMLLM, ModelConfig, load_checkpoint, save_checkpoint

REPO = pathlib.Path(__file__).resolve().parents[1]
EXP09 = REPO / "kaggle" / "experiments" / "09_instance_norm_retrain" / "kaggle_train_full.py"
ETTH1_CKPT = REPO / "kaggle" / "datasets" / "checkpoint" / "etth1_96_instnorm_best.pt"


class FakeBackbone(nn.Module):
    """Stands in for Llama: same (batch, channels, tokens, hidden) in and out, mixes tokens causally."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.tanh(self.proj(x.cumsum(dim=2) / torch.arange(1, x.shape[2] + 1).view(1, 1, -1, 1)))


def small_config(**kw):
    return ModelConfig(**{"patch_len": 24, "seq_len": 96, "mlp_dim": 16, "llm_dim": 32, "hidden_dim": 16, **kw})


@pytest.fixture
def model():
    torch.manual_seed(0)
    c = small_config()
    return FMLLM(c, FakeBackbone(c.llm_dim)).eval()


def test_config_rejects_history_not_multiple_of_patch():
    with pytest.raises(ValueError):
        ModelConfig(patch_len=96, seq_len=700)


def test_forward_shapes(model):
    pred, top_indices, scores = model(torch.randn(5, 3, 96))
    assert pred.shape == (5, 3, 24)
    assert top_indices.shape == (5, 3, 4, 1)
    assert scores.shape == (5, 3, 4, 2)


def test_instance_norm_makes_output_follow_input_level(model):
    """Shifting and scaling the whole history shifts and scales the forecast the same way."""
    x = torch.randn(2, 1, 96)
    a, _, _ = model(x)
    b, _, _ = model(x * 3 + 50)
    torch.testing.assert_close(b, a * 3 + 50, rtol=1e-4, atol=1e-3)


def test_forecast_single_step_equals_forward(model):
    x = torch.randn(2, 2, 96)
    direct, _, _ = model(x)
    torch.testing.assert_close(forecast(model, x, 24), direct)
    torch.testing.assert_close(forecast(model, x, 10), direct[..., :10])


def test_forecast_multi_step_length(model):
    assert forecast(model, torch.randn(1, 2, 96), 50).shape == (1, 2, 50)


def test_rollout_feeds_predictions_back():
    seen = []

    def predict_fn(ctx):
        seen.append(ctx.clone())
        return ctx[:, :, -1, :] + 1

    ctx = torch.zeros(1, 1, 3, 2)
    out = autoregressive_rollout(ctx, predict_fn, 3)
    assert out[0, 0, :, 0].tolist() == [1, 2, 3]
    assert seen[2][0, 0, :, 0].tolist() == [0, 1, 2]  # oldest token dropped, predictions appended


def test_matches_experiment_09_exactly():
    """Same weights -> same output as the original ETTh1 code: the refactor changed nothing."""
    spec = importlib.util.spec_from_file_location("exp09", EXP09)
    m09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m09)

    torch.manual_seed(0)
    c = ModelConfig()
    backbone = FakeBackbone(c.llm_dim)
    new = FMLLM(c, backbone).eval()
    old_emb = m09.FourierEmbeddingModule(patch_len=96, mlp_dim=512, llm_dim=2048).eval()
    old_dec = m09.FANMoEDecoder(llm_dim=2048, hidden_dim=512, patch_len=96, num_shared=2, num_routed=2, top_k=1).eval()
    old_emb.load_state_dict(new.embedding_module.state_dict())
    old_dec.load_state_dict(new.decoder.state_dict())

    history = torch.randn(2, 3, 672)
    with torch.no_grad():
        norm, mean, std = m09.instance_normalize(history)
        tokens = norm.unfold(-1, 96, 96)
        expected = m09.instance_denormalize(old_dec(backbone(old_emb(tokens)))[0][:, :, -1, :], mean, std)
        got, _, _ = new(history)
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not ETTH1_CKPT.exists(), reason="ETTh1 checkpoint not downloaded")
def test_loads_etth1_checkpoint():
    model = load_checkpoint(ETTH1_CKPT, FakeBackbone(2048))
    assert model.config.patch_len == 96 and model.config.num_tokens == 7


def test_checkpoint_roundtrip_keeps_config(tmp_path, model):
    save_checkpoint(model, tmp_path / "m.pt", epoch=3)
    loaded = load_checkpoint(tmp_path / "m.pt", model.llama_backbone)
    assert loaded.config == model.config
    x = torch.randn(1, 1, 96)
    torch.testing.assert_close(loaded(x)[0], model(x)[0])


def test_losses_are_finite_and_zero_at_target():
    t = torch.randn(4, 2, 24)
    assert hybrid_loss(t, t).item() == pytest.approx(0.0, abs=1e-6)
    assert hybrid_loss(t + 1, t).item() > 0
    scores = torch.softmax(torch.randn(4, 2, 4, 2), -1)
    top = scores.argmax(-1, keepdim=True)
    assert torch.isfinite(sequence_balance_loss(scores, top, num_routed=2, top_k=1))


def test_baselines():
    h = np.arange(200, dtype=float)[:, None]
    assert baseline_forecast(h, "persistence", 3)[:, 0].tolist() == [199, 199, 199]
    assert baseline_forecast(h, "daily_naive", 30)[:, 0].tolist() == list(range(176, 200)) + list(range(176, 182))
    assert baseline_forecast(h, "weekly_naive", 2)[:, 0].tolist() == [32, 33]


def test_assess_exceedance():
    r = assess_exceedance(np.array([10, 20, 40, 50, 30.0]), limit=35)
    assert r["exceeds"] and r["first_hour"] == 2 and r["hours_above"] == 2 and r["peak_hour"] == 3
    assert not assess_exceedance(np.array([1.0, 2.0]), limit=35)["exceeds"]
    assert assess_exceedance(np.array([33.0]), limit=35, margin=5)["exceeds"]


def test_scores_overall_and_per_location():
    recs = [
        {"location": "A", "event": True, "warned": True, "first_hour": 6},
        {"location": "A", "event": False, "warned": False},
        {"location": "B", "event": True, "warned": False, "first_hour": 3},
        {"location": "B", "event": False, "warned": True},
    ]
    s = score(recs)
    assert (s["hits"], s["misses"], s["false_alarms"], s["correct_negatives"]) == (1, 1, 1, 1)
    assert s["csi"] == pytest.approx(1 / 3) and s["mean_lead_hours"] == 6
    per = score_by(recs)
    assert per["A"]["hit_rate"] == 1.0 and per["B"]["hit_rate"] == 0.0


def test_forward_works_without_registered_parameters(model):
    """nn.DataParallel's per-GPU copies expose no parameters; forward must not depend on them."""
    x = torch.randn(2, 1, 96)
    expected, _, _ = model(x)
    model.decoder.parameters = lambda *a, **k: iter(())   # what a replica looks like
    model.embedding_module.parameters = lambda *a, **k: iter(())
    got, _, _ = model(x)
    torch.testing.assert_close(got, expected)


def test_set_decoder_dtype(model):
    model.set_decoder_dtype(torch.bfloat16)
    assert model.decoder_dtype == torch.bfloat16
    assert next(model.decoder.parameters()).dtype == torch.bfloat16
    pred, _, _ = model(torch.randn(2, 1, 96))
    assert pred.dtype == torch.float32 and torch.isfinite(pred).all()


def test_event_occurs_mean_vs_max():
    from fmllm.events import event_occurs
    v = np.array([10.0] * 23 + [100.0])            # one spike in an otherwise clean day
    assert event_occurs(v, "max", 24, 35.5)["exceeds"]
    assert not event_occurs(v, "mean", 24, 35.5)["exceeds"]   # mean = 13.75
    assert event_occurs(v, "mean", 24, 35.5)["value"] == pytest.approx(13.75)
    assert event_occurs(v, "max", 24, 35.5)["first_hour"] == 23
