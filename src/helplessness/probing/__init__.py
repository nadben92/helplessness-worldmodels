"""Torch-side measurement of a world model: the only place that runs forward passes through a frozen RSSM to
measure it (canonical probe, real-context probes, kernels in s-space, template matching, latent probes). It
returns plain arrays, which measures/ turns into controllability, diagnostics and RSA -- measures/ itself never
imports torch or a model.

probe.py  E3's probes (moved from experiments/E3_worldmodel_4zones/probe.py)

New probes (E4's fixed-trajectory context, E7's latent manipulations applied to a model) are new modules here.
"""
