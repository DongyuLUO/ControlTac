# v0.1.0 — Reconstructed ControlTac code release

This private prerelease contains the restored two-stage training/inference code, deterministic data manifests, two examples, and tensor-only historical weights.

Download `force_control.pth` and `force_pose_control.pth` into `checkpoints/`. Use the background-subtracted images already provided by [FeelAnyForce](https://huggingface.co/datasets/amirsh1376/FeelAnyForce). The `controltac_annotations.zip` supplement contains 1,800 aligned masks and pixel checksums, with no tactile images. Follow `docs/DATA.md` to link the upstream images and verify them. The prior 3.12 GB image bundle is not redistributed here.

Training counts: 20,000 force samples (19,824 distinct images plus 176 explicit repeats) and 7,000 distinct force+pose images with 300 unique poses per physical object. Full paper training and metrics have not been rerun; both historical inference examples and the two-stage one-step training pipeline have passed locally.

This repository and its assets remain private. Project licensing and recovery details are documented in the repository.

Weights use concise filenames; tensor contents and hashes are unchanged. All objects and both stages use the maintainer-confirmed shared normalization profile. Use the main branch for the updated shared-normalization workflow. The v0.1.0 tag preserves the original source snapshot. The existing project website and both repositories' commit histories are retained in ControlTac.
