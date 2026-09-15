# Inference examples

These are two different target conditions for the same measured Cross reference image. Both initial and target Fz are within [-10, -1] N.

| Example | Configuration | Input and target assets | Fz change (N) | Contact pose |
| --- | --- | --- | --- | --- |
| Force only | [force.json](force.json) | [assets/force](assets/force) | -2.433 to -8.638 | Unchanged |
| Force and pose | [force_pose.json](force_pose.json) | [assets/force_pose](assets/force_pose) | -2.433 to -7.957 | Position moves 4.79 mm |

The reference images are intentionally identical. Each folder has its own measured target image and reference/target masks. Force-only inference does not use a mask; force-and-pose inference uses `target_mask.npy` at the new contact position. `target_residual.png` is a measured comparison image, not a generated result or an inference input.

Run from the repository root:

```bash
python run.py infer --stage force
python run.py infer --stage force_pose
```

Generated images are saved as `outputs/force.png` and `outputs/force_pose.png`. Full signed force vectors are in the configuration files; measured source annotations and contact poses are in [provenance.json](provenance.json). Both examples use [normalization.json](normalization.json).
