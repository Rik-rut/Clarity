"""Vendored AMT (All-Pairs Multi-Field Transforms) frame interpolation.

Source: https://github.com/mcg-nku/amt (CVPR 2023).
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
Non-commercial use only. See LICENSE for details.

Internal imports were rewritten from the upstream absolute ``networks.*`` /
``utils.*`` packages to this ``video_upscaler.amt`` package. The network
config YAMLs (``cfgs/*.yaml``) reference the models via
``video_upscaler.amt.networks.<name>.Model``.
"""
