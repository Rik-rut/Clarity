"""Tests for processing profiles and profile -> model/engine resolution."""

from __future__ import annotations

import pytest

from video_upscaler.models import (
    PROFILES,
    default_profile,
    description_for_profile,
    model_for_profile,
    ncnn_args_for_profile,
    scale_for_model,
)

PROFILE_NAMES = {
    "2x_Clean", "2x_Light", "2x_Balanced", "2x_Deep", "2x_Faithful",
    "3x_Clean", "3x_Deep", "3x_Faithful",
    "4x_Clean", "4x_Deep", "4x_Faithful",
}


def test_profile_set() -> None:
    assert set(PROFILES) == PROFILE_NAMES


def test_profile_model_mapping() -> None:
    assert PROFILES["2x_Balanced"][0] == "up2x-latest-denoise2x.pth"
    assert PROFILES["2x_Clean"][0] == "up2x-latest-no-denoise.pth"
    assert PROFILES["2x_Light"][0] == "up2x-latest-denoise1x.pth"
    assert PROFILES["2x_Deep"][0] == "up2x-latest-denoise3x.pth"
    assert PROFILES["2x_Faithful"][0] == "up2x-latest-conservative.pth"
    assert PROFILES["3x_Deep"][0] == "up3x-latest-denoise3x.pth"
    assert PROFILES["4x_Faithful"][0] == "up4x-latest-conservative.pth"


def test_default_profile() -> None:
    assert default_profile() == "2x_Balanced"


def test_model_and_description_for_profile() -> None:
    assert model_for_profile("2x_Balanced") == "up2x-latest-denoise2x.pth"
    assert description_for_profile("2x_Balanced") == PROFILES["2x_Balanced"][1]


def test_scale_for_model() -> None:
    assert scale_for_model("up2x-latest-denoise2x.pth") == 2
    assert scale_for_model("up3x-latest-no-denoise.pth") == 3
    assert scale_for_model("up4x-latest-conservative.pth") == 4
    with pytest.raises(ValueError):
        scale_for_model("some_unknown_model.pth")


def test_ncnn_args_for_profiles() -> None:
    # noise level: -1 = no-denoise, 0 = conservative, 1..3 = denoise levels
    assert ncnn_args_for_profile("2x_Clean") == {"s": 2, "n": -1}
    assert ncnn_args_for_profile("2x_Light") == {"s": 2, "n": 1}
    assert ncnn_args_for_profile("2x_Balanced") == {"s": 2, "n": 2}
    assert ncnn_args_for_profile("2x_Deep") == {"s": 2, "n": 3}
    assert ncnn_args_for_profile("2x_Faithful") == {"s": 2, "n": 0}
    assert ncnn_args_for_profile("3x_Clean") == {"s": 3, "n": -1}
    assert ncnn_args_for_profile("4x_Deep") == {"s": 4, "n": 3}


def test_descriptions_non_empty() -> None:
    for profile, (_, description) in PROFILES.items():
        assert description.strip(), f"{profile}"
