import torch
import transformers


def test_torch_imports_and_runs():
    x = torch.randn(4, 96)
    layer = torch.nn.Linear(96, 512)
    assert layer(x).shape == (4, 512)


def test_transformers_importable():
    assert transformers.__version__
