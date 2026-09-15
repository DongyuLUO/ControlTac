# ControlTac

Force and contact-pose controlled tactile image generation from a reference tactile image.

[Paper](https://arxiv.org/abs/2505.20498) · [FeelAnyForce dataset](https://huggingface.co/datasets/amirsh1376/FeelAnyForce) · [Recovery notes](docs/RECOVERY.md)

This is a reconstructed release of the original experiment code. The supplied model tensors are unchanged. The reconstructed data splits and corrected training loop are new; they do not establish a reproduction of the paper's reported metrics.

## Install

Use Python 3.11 and install a matching PyTorch/torchvision build for your CUDA version, then:

```bash
pip install -e .
```

Tested locally with PyTorch 2.7.1+cu118, torchvision, diffusers 0.38.0 and timm 1.0.7. The DC-AE is downloaded from `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers` on first use. Add `--local-only` to use an existing cache. No W&B account is needed.

## Generate an example

Download the tensor-only weights below and save them in `checkpoints/`:

| File | Purpose | Size |
| --- | --- | ---: |
| [force_control.pth](https://github.com/DongyuLUO/ControlTac/releases/download/v0.1.0/force_control.pth) | Force control | 590 MB |
| [force_pose_control.pth](https://github.com/DongyuLUO/ControlTac/releases/download/v0.1.0/force_pose_control.pth) | Force and pose control | 892 MB |

```bash
python run.py infer --stage force
python run.py infer --stage force_pose
```

Outputs are written to `outputs/`. The two examples include a reference image, background, target force, normalization, and an aligned mask for pose control. Use `--device cpu` if CUDA is unavailable. The codec runs on CPU by default to reduce GPU memory use; on a larger GPU, add `--codec-device cuda`.

The examples use measured Cross samples with both initial and target Fz within **[-10, -1] N**:

| Example | Initial force [Fx, Fy, Fz] (N) | Target force [Fx, Fy, Fz] (N) | Contact pose |
| --- | --- | --- | --- |
| Force control | [-0.068, 0.031, -2.433] | [-0.180, 0.285, -8.638] | Unchanged |
| Force and pose control | [-0.068, 0.031, -2.433] | [0.022, 0.480, -7.957] | Position shifts by 4.79 mm; target mask changes |

Each example has its own reference and measured target residual image under `examples/assets/`. Reference masks are included for comparison; force-only inference does not consume a mask. Source annotations and exact contact poses are recorded in `examples/provenance.json`.

For your own inputs:

```bash
python -m controltac.infer --stage force_pose --checkpoint checkpoints/force_pose_control.pth --reference reference_residual.png --background background.png --mask contact_mask.npy --initial-force -0.068 0.031 -2.433 --target-force -0.15 0.2 -8 --normalization examples/normalization.json --normalization-key shared --output outputs/custom.png
```

The reference must be a **background-subtracted image stored with a 127 gray offset**, matching the original `tactile_nobg` files. RGB inputs are resized to 320×256. Forces are signed `[Fx, Fy, Fz]` in newtons; the model receives `target − initial`. A mask is a 2D NumPy array or grayscale image in the same sensor coordinates. It is encoded independently of image normalization.

When using newly trained weights, also pass `--model-config runs/force_pose/config.json --normalization runs/force_pose/normalization.json`. This selects the corrected backbone path instead of the compatibility path required by the historical CN checkpoint.

All objects and both stages use the same `shared` normalization profile. It is selected automatically; no object name or normalization key is needed. `--normalization` can specify a saved profile file.

## Data

The six objects are Cross, Slim Cylinder, Thin Cylinder, Medium Cylinder, Big Sphere, and Triple Cylinder. Thin Cylinder is one object, including all of its source recordings.

| Object | Force samples | Force+pose images | Unique contact poses |
| --- | ---: | ---: | ---: |
| Cross | 3,334 | 1,167 | 300 |
| Slim Cylinder | 3,333 | 1,167 | 300 |
| Thin Cylinder | 3,334 | 1,166 | 300 |
| Medium Cylinder | 3,333 | 1,167 | 300 |
| Big Sphere | 3,333 | 1,167 | 300 |
| Triple Cylinder | 3,333 | 1,166 | 300 |
| **Total** | **20,000** | **7,000** | **1,800** |

These quotas are realized in the checked-in split. Stage one contains **19,824 distinct images and 176 explicitly repeated samples**, to preserve the specified source allocation within Thin Cylinder. See `splits/report.json` for the complete audit. Neither 20,000 nor 7,000 is divisible by six: the integer allocation differs by at most one image between objects. Stage two never repeats an image.

Data paths are relative to `--data-root`, the directory containing `data_all/`. **Use the background-subtracted images already provided by FeelAnyForce.** We do not re-upload its image dataset. Our small annotation supplement supplies aligned contact masks and a pixel-checksum index for the exact image selection. See [data preparation](docs/DATA.md) for downloading the upstream archive, linking its images, and adding the masks.

```bash
python tools/link_feelanyforce.py --source-root /path/to/extracted/FeelAnyForce --annotations controltac_annotations.zip --output data
python tools/verify_data.py --data-root data
python run.py train --data-root data
```

Images are linked after pixel verification, so no second image copy is needed on the same disk. Use `--copy` only if links are unavailable. [Weights and annotation supplement](https://github.com/DongyuLUO/ControlTac/releases/tag/v0.1.0) are private release assets; sign in with an authorized GitHub account to download them. No new 3.12 GB tactile-image archive is required.

To reconstruct from the original local CSV tree:

```bash
python -m controltac.prepare --data-root /path/to/source_data
```

This fails clearly if there are too few unique images. Only if repeated **force** samples are intended, explicitly add `--allow-repeated-force-samples`. The supplied Thin Cylinder source allocation requires this option for the requested quota. All repetitions are deterministic and recorded; masks and force labels are never invented.

## Train

After preparing the manifests, normalization, and data files:

```bash
python tools/verify_data.py --data-root /path/to/source_data
python run.py train --data-root /path/to/source_data
```

This trains force control first, then initializes force+pose control from the new force weights. To train stages separately:

```bash
python -m controltac.train --config configs/force.json --data-root /path/to/source_data
python -m controltac.train --config configs/force_pose.json --data-root /path/to/source_data --initialize-from runs/force/model.pth
```

Both stages default to 75,000 optimizer steps, batch size 4, AdamW, cosine annealing, and `0.5 L1 + 0.5 MSE` noise-prediction loss. Learning rates are `1e-4 → 1e-5` for force and `1e-5 → 1e-6` for pose, following [Appendix A.1](https://arxiv.org/html/2505.20498v1#A1.SS1). Weight decay and clipping are recovered code choices, not specified by the paper.

The four input images form 16 reference/target pairs, as in the surviving training code. Force batches share a contact pose; pose batches share an object recording. Pair gradients accumulate in microbatches to limit memory. Each optimizer step still includes all 16 pairs. Set `--pair-microbatch 16` for maximum parallelism when memory permits.

The codec stays frozen. Stage two freezes the backbone and trains the six copied ControlNet blocks plus the mask projection. The default output is a tensor-only `model.pth`, accompanied by `config.json`, `normalization.json`, and `metrics.jsonl`. Add `--save-training-state` to keep a separate `training_state.pt` with optimizer, scheduler, scaler, and RNG state. Resume with `--resume path/to/training_state.pt` and the same configuration.

Use `--max-steps 1 --device cpu --local-only` for a smoke run. A full 75,000-step run was **not** performed during recovery. The paper used an RTX A5000; the local 4 GB RTX 2050 was used for inference, and CPU for training smoke tests.

## Layout

```text
controltac/       models, diffusion, data preparation, training, inference
configs/         two paper-aligned training configurations
examples/        two self-contained examples and provenance
splits/          portable manifests and reconstruction report
checkpoints/     tensor-only weights and SHA-256 manifest
tools/           verification, export, and release asset packaging
tests/           sampling, normalization, and DDIM regression tests
docs/            recovery decisions and data release instructions
```

```bash
python -m unittest discover -s tests -v
```

## Attribution and licensing

The implementation builds on SANA/DC-AE and DiT-related transformer components. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the preserved upstream licenses. Dataset derivatives retain FeelAnyForce attribution. The original workspace did not contain a project license; the maintainer must confirm the code's provenance and choose the project's license before publishing. This recovery does not silently relicense inherited code or checkpoints.
