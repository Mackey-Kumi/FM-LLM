import importlib.util
import pathlib

import pytest
import torch

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "kaggle"
    / "experiments"
    / "09_instance_norm_retrain"
    / "kaggle_train_full.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("instance_norm_train_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load_module()


def test_instance_normalize_denormalize_roundtrip(m):
    torch.manual_seed(0)
    x = torch.randn(4, 3, 672) * 10 + 5
    x_norm, mean, std = m.instance_normalize(x)
    x_recon = m.instance_denormalize(x_norm, mean, std)
    assert torch.allclose(x_recon, x, atol=1e-4)


def test_instance_normalize_output_shape_and_stats(m):
    torch.manual_seed(0)
    x = torch.randn(4, 3, 672)
    x_norm, mean, std = m.instance_normalize(x)
    assert x_norm.shape == x.shape
    assert mean.shape == (4, 3, 1)
    assert std.shape == (4, 3, 1)
    assert torch.allclose(x_norm.mean(dim=-1), torch.zeros(4, 3), atol=1e-4)
    assert torch.allclose(x_norm.std(dim=-1, unbiased=False), torch.ones(4, 3), atol=1e-2)


def test_instance_normalize_uses_each_windows_own_stats(m):
    torch.manual_seed(0)
    batch1 = torch.randn(1, 2, 96) * 3 + 50
    batch2 = torch.randn(1, 2, 96) * 0.1 - 20
    x = torch.cat([batch1, batch2], dim=0)  # (2, 2, 96): two independent windows

    x_norm, mean, std = m.instance_normalize(x)

    assert torch.allclose(mean[0], batch1.mean(dim=-1, keepdim=True), atol=1e-4)
    assert torch.allclose(mean[1], batch2.mean(dim=-1, keepdim=True), atol=1e-4)


def test_instance_denormalize_broadcasts_to_shorter_patch(m):
    torch.manual_seed(0)
    history = torch.randn(2, 3, 672) * 5 + 10
    _, mean, std = m.instance_normalize(history)

    patch_norm = torch.randn(2, 3, 96)
    patch_denorm = m.instance_denormalize(patch_norm, mean, std)

    assert patch_denorm.shape == (2, 3, 96)
    expected = patch_norm * std + mean
    assert torch.allclose(patch_denorm, expected)
