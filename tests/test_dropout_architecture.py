import importlib.util
import pathlib

import pytest
import torch
import torch.nn as nn

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "kaggle"
    / "experiments"
    / "05_dropout_retrain"
    / "kaggle_train_full.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("dropout_train_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    return _load_module()


class _IdentityBackbone(nn.Module):
    """Stands in for the frozen Llama backbone: preserves the (b, s, n, h)
    shape contract without loading real weights, so the pipeline shape flow
    can be tested on CPU with no GPU/HF token/network access."""

    def forward(self, embedding_inputs):
        return embedding_inputs


def test_fourier_embedding_module_output_shape(m):
    module = m.FourierEmbeddingModule(patch_len=96, mlp_dim=32, llm_dim=64)
    x = torch.randn(2, 3, 4, 96)
    out = module(x)
    assert out.shape == (2, 3, 4, 64)


def test_fan_expert_output_shape(m):
    expert = m.FANExpert(llm_dim=64, hidden_dim=32, patch_len=96)
    x = torch.randn(2, 3, 4, 64)
    out = expert(x)
    assert out.shape == (2, 3, 4, 96)


def test_routed_expert_has_no_dropout(m):
    expert = m.RoutedExpert(llm_dim=64, hidden_dim=32, patch_len=96)
    assert not any(isinstance(mod, nn.Dropout) for mod in expert.modules())


def test_embedding_module_has_one_dropout(m):
    embed = m.FourierEmbeddingModule(patch_len=96, mlp_dim=32, llm_dim=64)
    assert sum(isinstance(mod, nn.Dropout) for mod in embed.modules()) == 1


def test_fan_expert_has_two_dropouts(m):
    expert = m.FANExpert(llm_dim=64, hidden_dim=32, patch_len=96)
    assert sum(isinstance(mod, nn.Dropout) for mod in expert.modules()) == 2


def test_dropout_active_in_train_mode_inactive_in_eval(m):
    torch.manual_seed(0)
    embed = m.FourierEmbeddingModule(patch_len=96, mlp_dim=32, llm_dim=64, dropout=0.5)
    x = torch.randn(1, 1, 1, 96)

    embed.train()
    out1 = embed(x)
    out2 = embed(x)
    assert not torch.allclose(out1, out2), "dropout should make repeated train-mode forward passes differ"

    embed.eval()
    out3 = embed(x)
    out4 = embed(x)
    assert torch.allclose(out3, out4), "eval mode should be deterministic (dropout inactive)"


def test_full_pipeline_shape_with_dummy_backbone(m):
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
    embeds = embedding_module(tokens)
    llm_out = backbone(embeds)
    forecast, top_indices, scores = decoder(llm_out)
    pred_patch = forecast[:, :, -1, :]

    assert pred_patch.shape == (batch, channels, pred_len)
    assert top_indices.shape == (batch, channels, num_tokens, 1)
    assert scores.shape == (batch, channels, num_tokens, 2)
