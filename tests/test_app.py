import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app import inference
from app.api import main as api_main

REPO = pathlib.Path(__file__).resolve().parents[1]


class FakeBackbone(nn.Module):
    """Stands in for frozen Llama: identity on the embeddings."""

    def forward(self, embedding_inputs):
        return embedding_inputs


@pytest.fixture(scope="module")
def data():
    return inference.ETTh1Data()


@pytest.fixture
def live_engine(tmp_path, monkeypatch):
    """Engine with a randomly initialized instnorm checkpoint on disk and no precomputed file."""
    torch.manual_seed(0)
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    torch.save(
        {
            "embedding_module": inference.FourierEmbeddingModule().state_dict(),
            "decoder": inference.FANMoEDecoder().state_dict(),
        },
        ckpt_dir / inference.CHECKPOINT_SPECS["instnorm"][0],
    )
    monkeypatch.setattr(inference, "CHECKPOINT_DIR", ckpt_dir)
    return inference.ForecastEngine(backbone=FakeBackbone(), precomputed_path=tmp_path / "none.npz")


@pytest.fixture
def perfect_engine(tmp_path, monkeypatch, data):
    """Engine serving precomputed forecasts equal to the true future, no checkpoints."""
    monkeypatch.setattr(inference, "CHECKPOINT_DIR", tmp_path / "missing")
    windows = np.arange(0, data.num_windows, 240)
    truth = np.stack([
        data.test_norm[w + inference.SEQ_LEN: w + inference.SEQ_LEN + inference.MAX_HORIZON]
        for w in windows
    ])
    path = tmp_path / "forecasts.npz"
    np.savez(path, window_indices=windows, instnorm=truth)
    return inference.ForecastEngine(precomputed_path=path)


def test_split_matches_paper_and_local_csv(data):
    assert data.num_windows == 2161
    assert data.channel_names == inference.CHANNEL_NAMES
    stats = pd.read_json(REPO / "app" / "demo_data" / "norm_stats.json")
    np.testing.assert_allclose(data.channel_mean, stats["mean"][data.channel_names].values, rtol=1e-5)


def test_ground_truth_is_real_units_at_the_right_dates(perfect_engine):
    result = perfect_engine.forecast(0, 96, "instnorm")
    csv = pd.read_csv(inference.DATA_CSV).set_index("date")
    expected = csv.loc[result["dates"], result["channels"]].values
    np.testing.assert_allclose(result["ground_truth"], expected, rtol=1e-5)
    # History ends right where the forecast starts.
    assert pd.Timestamp(result["history_dates"][-1]) + pd.Timedelta(hours=1) == pd.Timestamp(result["dates"][0])


def test_perfect_forecast_scores_zero_and_matches_truth_scale(perfect_engine):
    result = perfect_engine.forecast(240, 720, "instnorm")
    assert result["source"] == "precomputed"
    assert result["metrics"]["mse"] == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(result["predictions"], result["ground_truth"], rtol=1e-4, atol=1e-4)


def test_precomputed_only_engine_exposes_only_precomputed_windows(perfect_engine):
    assert perfect_engine.live_checkpoints() == []
    assert perfect_engine.available_checkpoints() == ["instnorm"]
    assert perfect_engine.windows("instnorm") == list(range(0, 2161, 240))
    with pytest.raises(RuntimeError, match="Live inference"):
        perfect_engine.forecast(1, 96, "instnorm")


def test_backtest_uses_precomputed_windows(perfect_engine):
    out = perfect_engine.backtest("instnorm", num_windows=3, horizons=(96, 720))
    assert out["windows"] == [0, 960, 2160]
    assert out["results"][720]["num_windows"] == 3
    assert out["results"][96]["mse_mean"] == pytest.approx(0.0, abs=1e-10)


def test_live_forecast_matches_rollout(live_engine, data):
    result = live_engine.forecast(5, 336, "instnorm")
    assert result["source"] == "live"
    assert result["predictions"].shape == (336, 7)

    emb, dec, use_in = live_engine._models["instnorm"]
    history = torch.from_numpy(data.test_norm[5:5 + inference.SEQ_LEN].T.copy()).unsqueeze(0)
    expected = inference.rollout_forecast(history, emb, dec, FakeBackbone(), use_in)
    expected = data.denormalize(expected.squeeze(0).numpy().T[:336])
    np.testing.assert_allclose(result["predictions"], expected, rtol=1e-5, atol=1e-5)


def test_scenario_zero_change_equals_base_and_real_change_moves_forecast(live_engine):
    same = live_engine.scenario_forecast(3, 96, "instnorm", {"OT": 0.0})
    np.testing.assert_allclose(same["predictions"], same["base"], rtol=1e-5, atol=1e-5)
    changed = live_engine.scenario_forecast(3, 96, "instnorm", {"HUFL": 0.5})
    assert not np.allclose(changed["predictions"], changed["base"])


def test_api_endpoints(perfect_engine, monkeypatch):
    monkeypatch.setattr(api_main, "engine", perfect_engine)
    client = TestClient(api_main.app)  # no `with`: skip lifespan, use the injected engine

    health = client.get("/health").json()
    assert health["checkpoints"] == ["instnorm"] and health["live_checkpoints"] == []

    windows = client.get("/windows", params={"checkpoint": "instnorm"}).json()
    assert windows["windows"][:2] == [0, 240]

    r = client.post("/forecast", json={"window_idx": 0, "horizon": 192, "checkpoint": "instnorm"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 192 and len(body["dates"]) == 192
    assert body["metrics"]["mse"] == pytest.approx(0.0, abs=1e-10)

    assert client.post("/forecast", json={"window_idx": 0, "horizon": 100}).status_code == 400
    assert client.post("/forecast", json={"window_idx": 0, "horizon": 96, "checkpoint": "dropout"}).status_code == 400
    assert client.post("/forecast", json={"window_idx": 1, "horizon": 96}).status_code == 409

    r = client.post("/scenario", json={"window_idx": 0, "horizon": 96, "perturbations": {"OT": 0.1}})
    assert r.status_code == 409  # needs live inference

    r = client.post("/backtest", json={"num_windows": 2, "horizons": [96, 192]})
    assert r.status_code == 200 and set(r.json()["results"]) == {"96", "192"}

    demo = client.get("/demo/window/0").json()
    assert len(demo["history"]) == inference.SEQ_LEN and len(demo["future"]) == inference.MAX_HORIZON

    assert client.get("/results").json()["paper"]["720"] == [0.397, 0.429]


def test_kaggle_app_forecasts_kernel_imports_without_gpu():
    path = REPO / "kaggle" / "experiments" / "11_app_forecasts" / "kaggle_app_forecasts.py"
    spec = importlib.util.spec_from_file_location("app_forecasts_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "autoregressive_rollout") and hasattr(module, "instance_normalize")
