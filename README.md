## TractCloud

This repository releases the source code, training data, trained model, and testing samples for the work, "TractCloud: Registration-free tractography parcellation with a novel local-global streamline point cloud representation", which is accepted by MICCAI 2023.

![overview_v3](https://github.com/tengfeixue-victor/TractCloud-OpenSource/assets/56477109/1d41ef2c-367e-41dc-bfe2-6df955fc89d3)

## Inference CLI

TractCloud includes a pip-installable command-line tool for registration-free
tractography parcellation. It classifies streamlines from whole-brain
tractography into 42 anatomical white matter tracts using the pre-trained
DGCNN model, without requiring image registration to an atlas.

### Installation

```bash
pip install -e .
```

This installs the `tractcloud` CLI and Python package. Dependencies: `numpy`,
`torch`, and `vtk`. GPU (CUDA) is used automatically when available.

### Quick start

```bash
tractcloud --input brain_tractography.vtk --output-dir results/
```

This downloads the pre-trained model on first use (~50 MB), parcellates the
input tractography, and writes per-tract VTP files organized by anatomical
category:

```
results/
  Association/
    arcuate fasciculus (AF).vtp
    cingulum bundle (CB).vtp
    ...
  Projection/
    corticospinal tract (CST).vtp
    ...
  Commissural/
    corpus callosum 1 (CC1).vtp
    ...
  Cerebellar/
    ...
  Superficial/
    ...
```

### Options

```
tractcloud --input FILE --output-dir DIR [options]

  --mrb              Also create a Slicer-compatible MRB file with
                     SubjectHierarchy, colors, and display settings
  --include-other    Include 'Other' bundle for unclassified streamlines
  --device auto|cpu|cuda   Compute device (default: auto)
  --batch-size N     Inference batch size (default: 2048)
  --data-dir DIR     Override model data cache directory
  --quiet            Suppress JSON progress output on stdout
```

### MRB output

With `--mrb`, the tool creates a Slicer-loadable `.mrb` file containing all
tracts with unique colors organized in a two-level SubjectHierarchy
(category folders > individual tracts with full anatomical names).

### Python API

```python
from tractcloud import TractCloudPipeline

pipeline = TractCloudPipeline(device="auto")
result = pipeline.run_on_file("brain.vtk", "output/", create_mrb=True)
```

### Performance

On an RTX 5060 Ti (16 GB), 500,000 streamlines are parcellated in ~74 seconds.
The inference step alone takes ~33 seconds on GPU vs ~574 seconds on CPU (17x
speedup).

### 3D Slicer integration

The [SlicerDMRI](https://github.com/SlicerDMRI/SlicerDMRI) extension includes
a TractCloud module that provides a graphical interface for this tool within
3D Slicer.

---

## License

The contents of this repository are released under an [Slicer](LICENSE) license.

## Dependencies (training)

The environment test was performed on RTX4090 and A5000

`conda create --name TractCloud python=3.8`

`conda activate TractCloud`

`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`

`conda install -c fvcore -c iopath -c conda-forge fvcore iopath`

`conda install -c bottler nvidiacub`

`pip install pytorch3d`

`pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git`

`pip install h5py`

`pip install seaborn`

`pip install scikit-learn`

`pip install openpyxl`

## Training on anatomically curated atlas (ORG atlas)

The ORG atlas used in training is available at http://dmri.slicer.org/atlases/. You can directly download our processed data at https://github.com/SlicerDMRI/TractCloud/releases (1 million streamlines, 800 clusters & 800 outliers).
1. Download `TrainData_800clu800ol.tar.gz` to `./` and `tar -xzvf TrainData_800clu800ol.tar.gz`
2. Run `cd ./train_test && sh TrainOnAtlas.sh`

## Training on your custom dataset
Your input streamline features should have size of (number_streamlines, number_points_per_streamline, 3), and size of labels is (number_streamlines, ). You may save/load features and labels using .pickle files.

## Train/Validation/Test results and tips
The script calculates the accuracy and f1 on 42 anatomically meaningful tracts and one "Other" category (43 classes).

For training using the setting reported in our paper (k=20, k_global=500), most of CPU memory consumption comes from k. If you get out of CPU memory issue, you can try to reduce the value of k. Most of GPU memory consumption comes from k_global. If you get out of GPU memory issue, you can try to reduce the value of k_global.

## Testing on real data (registration-free parcellation)
Use the our trained model to parcellate real tractography data without registration.
1. Download `TrainedModel.tar.gz` (https://github.com/SlicerDMRI/TractCloud/releases) to `./`, and `tar -xzvf TrainedModel.tar.gz`
2. Download `TestData.tar.gz` (https://github.com/SlicerDMRI/TractCloud/releases) to `./`, and `tar -xzvf TestData.tar.gz`
3. Run `cd ./train_test && sh TractCloud.sh`

## Visualizing test parcellation results

Install 3D Slicer (https://www.slicer.org) and SlicerDMRI (http://dmri.slicer.org).

vtp/vtk files of 42 anatomically meaningful tracts are in `./parcellation_results/[test_data]/[subject_id]/SS/predictions`. "SS" means subject space. 

You can visualize them using 3D Slicer.

![TestExamples](https://github.com/SlicerDMRI/TractCloud/assets/56477109/efc55d90-4cf5-422c-9abe-ddb82c06dd6f)

## References

**Please cite the following papers for using the code and/or the training data:**
    
    Tengfei Xue, Yuqian Chen, Chaoyi Zhang, Alexandra J. Golby, Nikos Makris, Yogesh Rathi, Weidong Cai, Fan Zhang, Lauren J. O'Donnell 
    TractCloud: Registration-free Tractography Parcellation with a Novel Local-global Streamline Point Cloud Representation.
    International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023.

    Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, LJ. 
    An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. 
    NeuroImage, 2018 (179): 429-447
