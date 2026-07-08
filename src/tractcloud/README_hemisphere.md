# Hemisphere Export Mode

`tractcloud --hemisphere-atlas-dir PATH` runs TractCloud inference, keeps the
fine cluster predictions, registers the input tractography to the ORG atlas
with WMA, and writes hemisphere-split tractography outputs.

Default CLI behavior is unchanged when `--hemisphere-atlas-dir` is omitted.
Hemisphere mode writes only Hemisphere outputs, not the default coarse
per-tract category folders. `--mrb` is not supported in this mode.

## Requirements

- WMA must be installed on the machine running the command (pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git).
- `PATH` should point to `ORG-Atlases-1.1.1` (https://www.dropbox.com/s/beju3c0g9jqw5uj/WMA_tutorial_data.zip?dl=0) or a parent directory containing:

```text
ORG-Atlases-1.1.1/ORG-RegAtlas-100HCP/registration_atlas.vtk
ORG-Atlases-1.1.1/ORG-800FC-100HCP/cluster_hemisphere_location.txt
```

## Usage

```bash
tractcloud \
  --input subject.vtk \
  --output-dir results \
  --hemisphere-atlas-dir /path/to/ORG-Atlases-1.1.1
```

## Outputs

```text
results/
  wma_registration/
  Hemisphere/
    annotated_tractography.vtp
    hemisphere_summary.json
    left/
      AF_left.vtp
    right/
      AF_right.vtp
    commissural/
      CC1_commissural.vtp
```

All Hemisphere `.vtp` files include these cell arrays:

- `FineLabel`: raw TractCloud cluster prediction.
- `CoarseID`: tract index from `tract_mapping.py`.
- `HemisphereLocation`: `0=other`, `1=left`, `2=right`, `3=commissural`.
