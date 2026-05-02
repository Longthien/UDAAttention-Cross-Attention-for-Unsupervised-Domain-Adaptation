## TADA: Target-Augmented Domain Adaptation

**by Thien Bui Long**


## Overview

*Unsupervised Domain Adaptation (UDA)* for semantic segmentation is often hindered by spatial misalignment between domains and the prohibitive $\mathcal{O}(N^2)$ computational cost of high-resolution cross-attention. To solve this, we propose *Target-Augmented Domain Adaptation (TADA)*, a framework that explicitly bridges the domain gap by augmenting source features with semantically aligned target context before classification.
TADA achieves this via a Hybrid Cross-Attention Neck that dynamically processes features based on their spatial scale. It leverages highly efficient Deformable Cross-Attention at high resolutions to align local geometry and boundaries (bypassing the memory bottleneck), while utilizing standard Multi-Head Cross-Attention at low resolutions to capture global scene layout. Guided by a prototype-driven contrastive loss, TADA ensures that source queries accurately locate and extract semantically consistent target distributions, preventing feature collapse and significantly improving cross-domain generalization.

*Pushes SOTA Boundaries:* Integrating TADA into DAFormer yields a new peak performance of 69.87 mIoU, a +1.57 absolute improvement over the standard baseline.

For more information on DAFormer, please check the [original repository](https://github.com/lhoyer/DAFormer).

If you find this project useful in your research, please consider citing:

```
@InProceedings{
}

@Article{}

```

## Comparison with Baseline

| Class         | DAFormer w/o Fdist |  DAFormer | TADA + DAFormer w/o Fdist | TADA + DAFormer |
| ------------- | -----------------: | --------: | ------------------------: | --------------: |
| Road          |              95.88 |     95.70 |                     95.45 |       **97.04** |
| Sidewalk      |              71.01 |     70.20 |                     67.92 |       **77.00** |
| Building      |              89.40 |     89.40 |                 **89.62** |           89.13 |
| Wall          |              53.10 |     53.50 |                 **55.06** |           42.90 |
| Fence         |              45.08 | **48.10** |                     48.07 |           46.76 |
| Pole          |              49.50 |     49.60 |                     51.82 |       **53.29** |
| Traffic Light |              56.61 |     55.80 |                     58.38 |       **61.29** |
| Traffic Sign  |              59.38 |     59.40 |                     63.62 |       **66.31** |
| Vegetation    |          **90.01** |     89.90 |                 **90.04** |           89.93 |
| Terrain       |              49.94 |     47.90 |                     49.28 |       **50.09** |
| Sky           |              91.84 | **92.50** |                     91.93 |           92.16 |
| Person        |              71.45 |     72.20 |                     72.94 |       **74.80** |
| Rider         |              44.84 |     44.70 |                     45.43 |       **50.16** |
| Car           |              92.22 |     92.30 |                 **92.92** |           91.85 |
| Truck         |              68.85 |     74.50 |                 **77.93** |           67.71 |
| Bus           |              73.64 |     78.20 |                     76.19 |       **79.03** |
| Train         |              48.58 |     65.10 |                     55.29 |       **74.99** |
| Motorcycle    |              57.14 |     55.90 |                     55.54 |       **60.34** |
| Bicycle       |          **65.43** |     61.80 |                     63.29 |           62.65 |
| **mIoU**      |              67.05 |     68.30 |                     68.46 |       **69.87** |

## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/daformer
source ~/venv/daformer/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT ImageNet weights (b3-b5) provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.
Further, download the checkpoint of [DAFormer on GTA→Cityscapes](https://drive.google.com/file/d/1pG3kDClZDGwp1vSTEXmTchkGHmnLQNdP/view?usp=sharing) and extract it to the folder `work_dirs/`.


## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

**ACDC (Optional):** Please, download rgb_anon_trainvaltest.zip and
gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and
extract them to `data/acdc`. Further, please restructure the folders from
`condition/split/sequence/` to `split/` using the following commands:

```shell
rsync -a data/acdc/rgb_anon/*/train/*/* data/acdc/rgb_anon/train/
rsync -a data/acdc/rgb_anon/*/val/*/* data/acdc/rgb_anon/val/
rsync -a data/acdc/gt/*/train/*/*_labelTrainIds.png data/acdc/gt/train/
rsync -a data/acdc/gt/*/val/*/*_labelTrainIds.png data/acdc/gt/val/
```

**Dark Zurich (Optional):** Please, download the Dark_Zurich_train_anon.zip
and Dark_Zurich_val_anon.zip from
[here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it
to `data/dark_zurich`.

The final folder structure should look like this:

```none
DAFormer
├── ...
├── data
│   ├── acdc (optional)
│   │   ├── gt
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── dark_zurich (optional)
│   │   ├── gt
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia (optional)
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Training


A training job can be launched using:

```shell
python run_experiments.py --config configs/daformer/.py
```

For the experiments in our paper (e.g. network architecture comparison,
component ablations, ...), we use a system to automatically generate
and train the configs:

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [DAFormer](https://github.com/lhoyer/DAFormer.git)

## License

<!-- This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters. -->
