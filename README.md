# Adaptive Classifier-free Guidance for Robust Image-to-Image Translation

Official implementation of **AdaCFG**, an adaptive classifier-free guidance framework designed for robust image-to-image translation. AdaCFG predicts per-sample guidance scales and velocities, enabling stable edits across diverse target domains. The method is implemented on top of two editing backbones: **Plug-and-Play (PnP)** diffusion and **InstructPix2Pix (IP2P)**.

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)

---

## Installation

Create and activate the conda environment, then install the required packages.

```bash
conda create -n AdaCFG python=3.10 -y
conda activate AdaCFG
pip install -r requirements.txt
```

### Environment Variables (optional, for W&B logging)

Create a `.env` file in the project root:

```env
WANDB_API_KEY=<your_wandb_api_key>
WANDB_ENTITY=<your_entity>
WANDB_PROJECT=<your_project>
WANDB_MODE=online
```

---

## Dataset Preparation

### NuScenes

1. Download the dataset from the [official NuScenes website](https://www.nuscenes.org/).
2. Organize images into the following structure. Each split should contain the images used for training, validation, and testing respectively.

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

---

## Project Structure

```
AdaCFG/
├── configs/                      # YAML / JSON configuration files
│   ├── config.yaml               # PnP training config
│   ├── ip2p_config.yaml          # IP2P training config
│   ├── conditions.json           # Domain-descriptive prompts (CLIP-side)
│   ├── ip2p_conditions.json      # Instruction-style prompts (IP2P-side)
│   └── training_conditions.json  # Prompts used during training
├── data/                         # Dataset classes
├── models/                       # Guidance prediction models
├── util/                         # Losses, pipelines, metrics, schedulers
│   ├── pnp.py                    # PnP pipeline
│   ├── ip2p.py                   # InstructPix2Pix pipeline
│   ├── loss.py                   # Training objectives
│   ├── guidance_scheduler.py     # Adaptive guidance scheduler
│   └── metric.py                 # CLIP / DINO evaluation metrics
├── pnp_make_merge_data.py        # Preprocess latents/embeddings for PnP
├── ip2p_make_merge_data.py       # Preprocess embeddings for IP2P
├── pnp_train.py                  # Train AdaCFG (PnP backbone)
├── ip2p_train.py                 # Train AdaCFG (IP2P backbone)
├── pnp_main.py                   # Inference (PnP backbone)
└── ip2p_main.py                  # Inference (IP2P backbone)
```

---

## Preprocessing

Precompute the CLIP embeddings (and PnP latents) required for training.

### PnP backbone

```bash
python pnp_make_merge_data.py \
    --augmented_prompt_path configs/training_conditions.json \
    --image_data image_data \
    --latents_steps 50
```

### IP2P backbone

```bash
python ip2p_make_merge_data.py \
    --augmented_prompt_path configs/conditions.json \
    --ip2p_augmented_prompt_path configs/ip2p_conditions.json \
    --image_data image_data
```

Preprocessed files are saved under `merged_latents_forwards/`.

---

## Training

### PnP backbone

```bash
python pnp_train.py --config configs/config.yaml
```

### IP2P backbone

```bash
python ip2p_train.py --config configs/ip2p_config.yaml
```

Training logs are sent to Weights & Biases, intermediate samples are saved to `Train_images_results/<timestamp>/`, and checkpoints are saved to `ckpts/`.

---

## Inference

### PnP backbone

```bash
python pnp_main.py \
    --model_path ckpts/<your_checkpoint>.pt \
    --model_config configs/config.yaml \
    --prompt "A photo of a street at night." \
    --image_path <path_to_source_image> \
    --augmented_prompts configs/conditions.json \
    --save_path outputs/
```

### IP2P backbone

```bash
python ip2p_main.py \
    --model_path ckpts/<your_checkpoint>.pt \
    --model_config configs/ip2p_config.yaml \
    --prompt "A photo of a street at night." \
    --image_path <path_to_source_image> \
    --augmented_prompts configs/conditions.json \
    --ip2p_augmented_prompts configs/ip2p_conditions.json \
    --save_path outputs/
```

The target domain is automatically selected from `augmented_prompts` by cosine similarity with the input prompt, and the best candidate image is chosen based on a combined CLIP + DINO score.

---

## Configuration

Key fields in `configs/config.yaml` (PnP) and `configs/ip2p_config.yaml` (IP2P):

| Field | Description |
| --- | --- |
| `seed`, `device` | Random seed and compute device |
| `train_data_root`, `eval_data_root` | Image directories for train/valid splits |
| `train_embedding_data`, `eval_embedding_data` | Precomputed embeddings from the preprocessing step |
| `train_latent_data`, `eval_latent_data` | Precomputed latents (PnP only) |
| `batch_size`, `learning_rate`, `lr_lambda`, `epoch(s)` | Standard training hyperparameters |
| `model.init_g` | Initial guidance scale |
| `model.divide_out` | Output scaling factor for the guidance prediction head |
| `model.num_guidance_info` | Number of guidance outputs (e.g., init + velocity) |
| `model.hidden_dim`, `model.num_layers`, `model.length` | Guidance model architecture |
| `loss.negative_prompt` | Negative prompt used for classifier-free guidance |
| `loss.lambda_text / lambda_structure / lambda_mean / lambda_negative` | Loss term weights |
| `loss.pnp_injection_rate`, `loss.pnp_res_injection_rate` | PnP feature/residual injection rates |
| `loss.image_guidance`, `loss.devide_guide` | IP2P-specific guidance parameters |
| `loss.gradient`, `loss.schedule_method` | Guidance schedule (`decrease`, `cosine`, ...) |
| `loss.n_timestep`, `loss.latents_steps` | Diffusion steps and number of saved latent steps |
