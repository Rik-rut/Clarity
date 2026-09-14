"""Desktop shell support: first-run provisioning served over HTTP.

Everything in this package must stay importable with the standard library
only. It runs on the uv-managed interpreter *before* the project
dependencies (torch, opencv, fastapi) exist, so nothing here may import
them directly or through ``video_upscaler`` submodules.
"""
