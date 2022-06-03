import torch
import yaml

from iSeparate.models.d3net.d3net import MMD3Net
from iSeparate.models.demucs.demucs import Demucs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_d3net():
    with open("configs/d3net/vocals.yaml", "r") as ff:
        cfg = yaml.safe_load(ff)
    model = MMD3Net(**cfg["model_args"]).to(DEVICE)

    dummy_inp = torch.randn(2, 2, 6 * 44100).to(DEVICE)
    dummy_target = torch.randn(2, 1, 2, 6 * 44100).to(DEVICE)

    output, target = model(dummy_inp, dummy_target)

    assert output.shape == target.shape, "D3Net: Network input and output shapes should match"


def test_demucs():
    with open("configs/demucs/demucs_config.yaml", "r") as ff:
        cfg = yaml.safe_load(ff)
    model = Demucs(**cfg["model_args"]).to(DEVICE)

    dummpy_input = torch.randn(2, 2, 10 * 44100).to(DEVICE)
    dummpy_target = torch.randn(2, 4, 2, 10 * 44100).to(DEVICE)

    output, target = model(dummpy_input, dummpy_target)
    assert output.shape == target.shape, "Demucs: Network input and output shapes should match"
