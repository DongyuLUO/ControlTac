# Use FeelAnyForce directly

Download the existing **background-subtracted tactile images** from [amirsh1376/FeelAnyForce](https://huggingface.co/datasets/amirsh1376/FeelAnyForce). They are already part of the source dataset; ControlTac does not need to redistribute them. The dataset card specifies CC BY 4.0. Keep that attribution.

## 1. Download and extract the upstream dataset

The upstream dataset is stored as multipart archives, not as individually downloadable object folders. Its official extraction instructions use the `dataset.z01`, `dataset.z02`, `dataset.z03`, and `dataset.zip` volumes:

```bash
hf download amirsh1376/FeelAnyForce --repo-type dataset --include "dataset.z01" "dataset.z02" "dataset.z03" "dataset.zip" --local-dir feelanyforce_download
cd feelanyforce_download
zip -s 0 dataset.zip --out merged.zip
unzip merged.zip
```

These commands follow the [upstream extraction guide](https://huggingface.co/datasets/amirsh1376/FeelAnyForce/blob/main/README.md). On Windows use an archive tool with split-ZIP support, or run extraction in WSL. This volume set is about 82 GB; do not also download the alternative `dataset_part_*` files. Once extracted, only selected images are linked to training.

## 2. Add ControlTac annotations and link images

Download `controltac_annotations.zip` from the [private v0.1.0 release](https://github.com/DongyuLUO/ControlTac/releases/tag/v0.1.0). It contains aligned contact masks and `image_index.json`, **no tactile images**. Sign in to GitHub with an authorized account to access private assets.

From the ControlTac repository:

```bash
python tools/link_feelanyforce.py --source-root /path/to/extracted/dataset --annotations controltac_annotations.zip --output data
python tools/verify_data.py --data-root data
python run.py train --data-root data
```

The adapter discovers the six `object/tactile_nobg/` directories under the provided root. It checks original image dimensions and decoded BGR pixel SHA-256 against the reconstruction index, then links files into the `data_all/...` layout referenced by our CSVs. PNG metadata differences do not cause a mismatch. Raw images, missing images and incorrect pixels cause an explicit error. It does not subtract the background a second time or alter images to make a checksum pass.

Use a source root with one copy of each object. On the same filesystem the adapter creates hard links without duplicating image data; it falls back to symbolic links if supported. If neither is possible, use `--copy`. Linked files share the underlying source, so treat both trees as read-only.

The adapter is validated against the local extracted `tactile_nobg` tree used to reconstruct the splits. The full remote 82 GB archive was not downloaded again during publication. Its extracted pixels are verified when each user runs the adapter; a dataset-version mismatch fails instead of silently changing the experiment.

## Splits and annotations

Object labels are Cross, Slim Cylinder, Thin Cylinder, Medium Cylinder, Big Sphere, and Triple Cylinder. Thin Cylinder is one object.

Use the checked-in split CSVs and shared normalization for training. Force-only columns are `object`, `image`, `force`, and `reference_image`; images sharing a reference form same-contact pairs. Force-and-pose columns are `object`, `image`, `force`, and `mask`. Numeric poses and source bookkeeping are not included. Stage one contains 20,000 samples (19,824 unique images); stage two contains 7,000 unique images.

The new validation/test records hold out image, force and pose information. Mask fields can be empty because those poses were excluded from the aligned training masks. Supply independent aligned masks before evaluating mask-conditioned generation on them. New holdouts are not proven unseen data for historical checkpoints.

To regenerate the selection from the original source data, run `python -m controltac.prepare --data-root /path/to/source_data --allow-repeated-force-samples`. The preparation code checks pose counts and split separation in memory before writing training CSVs.

## Maintainer packaging

Source data can live in any directory; no `FT` directory name or repository-parent layout is required. The upstream CSV column `FT` means force/torque and is retained only when reading original annotations. Published manifests use the explicit `force` column.

Original checkpoint filenames are also unrestricted:

```bash
python tools/export_checkpoints.py --force-checkpoint /path/to/original_force_checkpoint.pth --force-pose-checkpoint /path/to/original_force_pose_checkpoint.pth --output checkpoints
python tools/make_examples.py --source-root /path/to/source_data
```

```bash
python tools/package_annotations.py --data-root /path/to/source_data
```

Upload the resulting annotation ZIP and the two tensor-only checkpoints to GitHub Releases. The older `tools/package_data.py` remains an optional offline-bundle utility; its 3.12 GB image archive is not part of this publication.
