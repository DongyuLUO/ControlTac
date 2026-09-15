# Recovery record

## Verified checkpoint mapping

- `force_control.pth` → `checkpoints/force_control.pth` (196 tensors).
- `force_pose_control.pth` → `checkpoints/force_pose_control.pth` (302 tensors; six ControlNet blocks).

The original files are untouched. Export strips epoch, optimizer, scheduler and scaler state without changing tensor values, names or precision. `checkpoints/manifest.json` contains sizes, SHA-256 hashes and exact equality results. The historical `phase_2` filename suffix denotes an optimizer schedule phase, not the paper's second component.

## Recovered architecture

The surviving model code and strict checkpoint loading establish 12 backbone blocks, width 768, 12 cross-attention heads, linear self-attention, GLUMBConv feed-forward layers, 64 concatenated latent input channels, and 32 output channels. Images have height 256 and width 320; DC-AE latents have height 8 and width 10. The historical positional-embedding base size `(10, 8)` is retained to preserve checkpoint behavior.

`model/net/FT_MS_F3.py`, `FT_MS_F3_Con.py` and their required modules were migrated into clearly named `controltac/models/` modules. Parameter names remain stable. No unrelated downstream estimators or experiment variants are imported.

## Corrections

- CLI parsing is inside `main`; importing a module does not start training.
- Configuration and sampling use only the requested six physical objects.
- Training schedules use 75,000 optimizer steps per stage and a single cosine decay.
- Training and inference share the same residual-image convention and saved normalization.
- Normalization for a new run is computed from unique first-stage training images only, shared with stage two, and saved next to weights. Test pixels cannot influence these statistics.
- Frozen DC-AE encoding runs without gradients and on a configurable device.
- Gradient clipping runs after AMP unscaling, and every pair contributes to an optimizer step.
- DDIM uses the 1,000-step training schedule independently of inference step count; the terminal alpha is 1, returning a clean latent. This corrects the old off-by-one sampler.
- New ControlNet training restores backbone block 0 and trains the mask projection. Existing CN weights use `legacy_skip_first_block=true`, retaining the historical forward traversal. Model class names changed; weight keys did not.
- Module calls use PyTorch's normal hooks instead of overriding `__call__`.
- Weight loading is strict, CPU-portable, and tensor-only. Full resumable state is an explicit separate training artifact.

## Data reconstruction and limits

This is a new deterministic split, not the missing original split. The original source files and their hashes are recorded. Primary CSVs are supplemented from the corresponding full local object CSV when necessary. The cylinder92_1 / cylinder92_2 labels come from their explicit source CSVs, not the misleading `train1` / `train2` filenames.

All annotated candidate training poses are excluded from new validation/test selection. Up to 30 distinct poses per physical object are used for each holdout. Both training stages may share poses and images, which is intentional for sequential training. Train/validation/test pose sets are disjoint. The holdout is newly reconstructed and must not be presented as an unseen evaluation set for the historical checkpoints, whose original training membership is unknown.

The force-pose reconstruction uses `*_train_pos_300.csv`; cylinder92_1 instead uses the surviving `*_train_pos_200.csv` to select 150 poses. Cylinder92_2 supplies another 150 poses, excluding numeric pose duplicates already selected from cylinder92_1. This yields 300 unique poses for the combined object, not 600. Selected images retain their original force and mask annotations.

The complete cylinder92_1 source has only 1,517 unique images; after the current holdout, 1,491 remain. A quota of 1,667 therefore needs 176 repeated samples. The strict preparation command refuses to fill this gap silently. Repetition, when explicitly selected, is confined to the force sample list and never to the 7,000-image pose list.

## Historical preprocessing uncertainty

The maintainer confirmed that all objects and both stages use the same normalization values. The fixed profile is named `shared`, retaining the recovered values unchanged. Preparation copies this profile; training and inference use it for every object. Earlier reconstruction versions computed per-subset statistics; that behavior has been corrected. The numerical paper results have not been rerun.

## Validation

See `docs/validation.json` for machine-readable outcomes. The recovery checks include exact tensor equality, strict checkpoint loading, both real checkpoint inference examples, a full optimizer step with 16 pairs for both stages, and regression tests for normalization, data selection and DDIM's clean terminal step. Complete model training and the paper's benchmark evaluation were not rerun.
