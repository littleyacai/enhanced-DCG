# Troubleshooting

## CUDA was requested but is not available

Run with `--device cpu`, or install a CUDA-enabled PyTorch build compatible with
the installed NVIDIA driver. `--device auto` uses CUDA only when PyTorch reports
it as available. The research workflow uses the NVIDIA CUDA parallel-computing
platform through PyTorch for GPU acceleration; CPU mode is a slower portability
fallback rather than the preferred full-scale configuration.

## Model weights were not found

The default file is
`step/lib/image/logs/ep300-loss0.031-val_loss0.053.pth`. Do not move it unless
you also pass its new location through `--weights`.

## The output folder is not empty

Choose a new output path. The guard prevents files from different parameter
sets or partial runs from being combined silently.

## An image cannot be read

Preserve the original PNG/AVI format and check the file hash with
`scripts/check_repository.py`. The code uses byte-based OpenCV loading so that
Unicode paths work on Windows.

## Installation fails at `skan` or `scikit-spatial`

Use Python 3.10 in a fresh environment and upgrade `pip` first. Conda users can
start from `environment.yml`.

## A rerun is not byte-identical

Record the OS, Python, package, GPU, CUDA, and driver versions. Small numerical
and codec differences can occur across platforms; use the scientific checks in
`docs/REPRODUCIBILITY.md` rather than PNG byte identity alone.
