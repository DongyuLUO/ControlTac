# Recovery record

## Pretrained checkpoints

Download `force_control.pth` and `force_pose_control.pth` from the [model release](https://github.com/DongyuLUO/ControlTac/releases/tag/v0.1.0) into `checkpoints/`. Both files contain only model parameters and required buffers. Sizes, tensor counts and SHA-256 checksums are recorded in `checkpoints/manifest.json`.

## Recovered architecture

The surviving model code and strict checkpoint loading establish 12 backbone blocks, width 768, 12 cross-attention heads, linear self-attention, GLUMBConv feed-forward layers, 64 concatenated latent input channels, and 32 output channels. Images have height 256 and width 320; DC-AE latents have height 8 and width 10. The historical positional-embedding base size `(10, 8)` is retained to preserve checkpoint behavior.

The original force and force-pose transformer implementations were migrated into `controltac/models/force_control.py` and `controltac/models/force_pose_control.py`, with a shared `diffusion_transformer.py` backbone. Parameter names remain stable. No unrelated downstream estimators or experiment variants are imported.

## Corrections

- CLI parsing is inside `main`; importing a module does not start training.
- Configuration and sampling use only the requested six physical objects.
- Training schedules use 75,000 optimizer steps per stage and a single cosine decay.
- Training and inference share the same residual-image convention and saved normalization.
- All objects and both stages use the fixed shared normalization profile, saved next to trained weights.
- Frozen DC-AE encoding runs without gradients and on a configurable device.
- Gradient clipping runs after AMP unscaling, and every pair contributes to an optimizer step.
- DDIM uses the 1,000-step training schedule independently of inference step count; the terminal alpha is 1, returning a clean latent. This corrects the old off-by-one sampler.
- New ControlNet training restores backbone block 0 and trains the mask projection. Existing CN weights use `legacy_skip_first_block=true`, retaining the historical forward traversal. Model class names changed; weight keys did not.
- Module calls use PyTorch's normal hooks instead of overriding `__call__`.
- Weight loading is strict, CPU-portable, and tensor-only. Full resumable state is an explicit separate training artifact.

## Data reconstruction and limits

This is a new deterministic split, not the missing original split. Primary CSVs are supplemented from the corresponding full local object CSV when necessary. Thin Cylinder is a single object; original recording identifiers are retained only in source provenance and file paths.

All annotated candidate training poses are excluded from new validation/test selection. Up to 30 distinct poses per physical object are used for each holdout. Both training stages may share poses and images, which is intentional for sequential training. Train/validation/test pose sets are disjoint. The holdout is newly reconstructed and must not be presented as an unseen evaluation set for the historical checkpoints, whose original training membership is unknown.

The force-pose reconstruction selects 300 unique poses for each of the six objects. Thin Cylinder combines its available annotated recordings, excluding numeric pose duplicates across recordings. Selected images retain their original force and mask annotations.

To preserve the original source allocation within Thin Cylinder, one recording contributes a quota of 1,667 samples from 1,491 available images after holdout, requiring 176 repeated samples. The strict preparation command refuses to fill this gap silently. Repetition, when explicitly selected, is confined to the force sample list and never to the 7,000-image pose list.

## Historical preprocessing uncertainty

The maintainer confirmed that all objects and both stages use the same normalization values. The fixed profile is named `shared`, retaining the recovered values unchanged. Preparation copies this profile; training and inference use it for every object. Earlier reconstruction versions computed per-subset statistics; that behavior has been corrected. The numerical paper results have not been rerun.

## Validation

The recovery checks include exact tensor equality, strict checkpoint loading, both real checkpoint inference examples, a full optimizer step with 16 pairs for both stages, and regression tests for normalization, data selection and DDIM's clean terminal step. Complete model training and the paper's benchmark evaluation were not rerun.
