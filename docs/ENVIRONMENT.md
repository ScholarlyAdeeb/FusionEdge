# Environment

Everything below was verified on this machine before being written down, so a
clone on an identical setup behaves the same and a clone on a different setup
knows what to expect.

## Hardware and software (measured, not assumed)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 **Laptop GPU**, 8 GiB VRAM |
| Driver | 616.92 (CUDA UDA 13.4) |
| Compute capability | 8.9 |
| OS | Windows (PowerShell) |
| Python | 3.14.5 |
| PyTorch | 2.12.1+cu126, `torch.cuda.is_available() == True` |
| torchvision | 0.27.1+cu126 |
| numpy / opencv / scikit-image | 2.5.0 / 5.0.0.93 / 0.26.0 |

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.12.1+cu126 True NVIDIA GeForce RTX 4070 Laptop GPU
```

## Python 3.14 compatibility

Python 3.14 is new enough that several standard CV packages have no wheels for
it. Every dependency was checked *before* being added, with
`pip install --dry-run`, rather than discovered at import time:

| Package | Status | Decision |
|---|---|---|
| `torch`, `torchvision` | pre-installed, CUDA builds working | use |
| `numpy`, `opencv-python`, `scikit-image`, `scipy`, `pillow`, `PyYAML`, `tqdm`, `matplotlib`, `tensorboard`, `pytest` | pre-installed | use |
| `ultralytics 8.4.175` | resolves on 3.14 | **primary** YOLO mAP backend |
| `onnxruntime 1.30.0` | already installed system-wide | **documented fallback** YOLO backend |
| `gradio 6.30.0` | resolves on 3.14 | demo |
| `gdown 6.4.2` | resolves on 3.14 | dataset download |

### Deliberate non-dependencies

`kornia` and `piq` were **not** adopted, even though they resolve on 3.14.
Every metric and loss term is implemented in this repository
(`fusionedge/metrics`, `fusionedge/losses`) and cross-validated in tests:

* SSIM is checked against `skimage.metrics.structural_similarity` to <1e-6,
* EN, AG, SF and MI are checked against literal re-implementations of their
  published definitions,
* MS-SSIM has no reference implementation in scikit-image, so it is validated by
  its defining properties (identity = 1, bounded in [0,1], monotone in distortion
  magnitude, symmetric).

The reason is defensibility: a fusion paper's numbers are only meaningful if you
can state exactly what each metric computes, and "we called a library" is not an
answer.

## Memory strategy for 8 GiB VRAM

| Setting | Value | Rationale |
|---|---|---|
| Train crop | 128x128 | dominates activation memory |
| Train batch | 8 (config knob) | lower first on OOM |
| Mixed precision | `torch.autocast("cuda", float16)` + `GradScaler` | ~half activation memory |
| Losses | computed in fp32 even under autocast | the intensity and gradient terms subtract near-identical tensors; fp16 loses exactly the small differences they measure |
| Memory format | `channels_last` | faster convs on tensor cores |
| Eval batch | 1, full image | batching buys nothing at 480x640 and costs VRAM |
| Eval data | streamed from disk | never all-in-RAM |
| Gradient clipping | 1.0 (config knob) | the losses contain `relu`/softmax terms that can spike |
| Non-divisible sizes | reflect-pad to a multiple of 8, crop back | fully-convolutional models handle any size; padding avoids artificial border edges |

## Reproducibility

* `deterministic: true` in the training config forces cuDNN determinism and
  disables TF32 fast paths. This costs ~10-20% throughput, which is the right
  trade for an ablation table whose only intended difference is the
  architecture.
* **Latency benchmarks turn determinism off** (and set `cudnn.benchmark = True`),
  because a latency number should reflect the fastest available kernels.
* Checkpoints store python/numpy/torch RNG state, so `--resume` continues the
  augmentation stream as well as the weights.
* Every run stamps its config with git commit, torch/CUDA versions, GPU name and
  the dataset id + manifest hash.

## Reproducing the environment

```powershell
python -m pip install -r requirements.txt          # core, nothing new on this machine
python -m pip install -r requirements-extras.txt   # ultralytics, gradio, gdown
python -m pytest tests -q                          # expect: all tests pass
```