# Adaptive Classifier-free Guidance for Robust Image-to-image translation


## Installation
First, create and activate the conda environment for AdaCFG, then install the required packages using the commands below.
```
conda create -n AdaCFG python=3.10 -y
conda activate AdaCFG
pip install -r requirements.txt
```



## Dataset Preparation
### NuScenes
1. Download the dataset from the [official NuScenes website](https://www.nuscenes.org/).
2. Organize the downloaded images into the following directory structure. As described in the paper, each folder should contain the images for training, validation, and testing.
```
image_data/
├── train/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── valid/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── test/
    ├── 0000.png
    ├── 0001.png
    └── ...
```