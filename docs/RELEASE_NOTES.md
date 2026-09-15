# v0.1.0 — Reconstructed ControlTac code release

This private prerelease contains the restored two-stage training/inference code, deterministic data manifests, two examples, and tensor-only historical weights.

Download `Only_Force_00_B_phase_2_checkpoint_epoch_65.pth` and `CN_300_00_phase_2_checkpoint_epoch_60.pth` into `checkpoints/`. Use the background-subtracted images already provided by [FeelAnyForce](https://huggingface.co/datasets/amirsh1376/FeelAnyForce). The `controltac_annotations.zip` supplement contains 1,800 aligned masks and pixel checksums, with no tactile images. Follow `docs/DATA.md` to link the upstream images and verify them. The prior 3.12 GB image bundle is not redistributed here.

Training counts: 20,000 force samples (19,824 distinct images plus 176 explicit repeats) and 7,000 distinct force+pose images with 300 unique poses per physical object. Full paper training and metrics have not been rerun; both historical inference examples and the two-stage one-step training pipeline have passed locally.

This repository and its assets remain private. Project licensing and historical CN preprocessing limitations are documented in the repository.

Asset filenames were restored to the author's original filenames on 2026-09-15; tensor contents and hashes are unchanged. The main branch examples use these names. The original v0.1.0 source tag retains its historical short filenames; when using that tag, provide `--checkpoint` explicitly with the filename shown above.
