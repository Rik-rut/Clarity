"""Optional fast-path dependency must not be required to import the app."""

from __future__ import annotations


def test_cli_imports_without_triton(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    from video_upscaler import cli, config, models, scanner  # noqa: F401

    assert config.MODELS_DIR is not None
