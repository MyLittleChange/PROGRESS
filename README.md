# Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection



[![Paper](https://img.shields.io/badge/paper-arxiv.2412.03561-B31B1B.svg)](https://arxiv.org/abs/2506.01085)
[![PROGRESS](https://img.shields.io/badge/Project-Page-FFD700?style=for-the-badge?logo=flag)](https://mylittlechange.github.io/PROGRESS_web/)


<div style="font-family: charter; text-align: center; margin-top: 2rem;">
    <a href="https://mylittlechange.github.io/" target="_blank" style="color: #333; font-size: 1.4rem; font-weight: bold; text-decoration: none; transition: color 0.3s; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        Qian Yang*
    </a>
    <span style="margin: 0 0.5rem; color: #666;">•</span>
    <a href="https://scholar.google.com/citations?hl=en&user=ZER2BeIAAAAJ&view_op=list_works&sortby=pubdate" target="_blank" style="color: #333; font-size: 1.4rem; font-weight: bold; text-decoration: none; transition: color 0.3s; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        Shivam Chandhok*
    </a>
    <span style="margin: 0 0.5rem; color: #666;">•</span>
    <a href="https://oscmansan.github.io/" target="_blank" style="color: #333; font-size: 1.4rem; font-weight: bold; text-decoration: none; transition: color 0.3s; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        Oscar Mañas
    </a>
    <span style="margin: 0 0.5rem; color: #666;">•</span>
    <a href="https://kanji95.github.io/" target="_blank" style="color: #333; font-size: 1.4rem; font-weight: bold; text-decoration: none; transition: color 0.3s; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        Kanishk Jain
    </a>
    <span style="margin: 0 0.5rem; color: #666;">•</span>
    <a href="https://www.cs.ubc.ca/~lsigal/" target="_blank" style="color: #333; font-size: 1.4rem; font-weight: bold; text-decoration: none; transition: color 0.3s; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        Leonid Sigal
    </a>
    <span style="margin: 0 0.5rem; color: #666;">•</span>
    <a href="https://www.iro.umontreal.ca/~agrawal/" target="_blank" style="color: #333; font-size: 1.4rem; font-weight: bold; text-decoration: none; transition: color 0.3s; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
        Aishwarya Agrawal
    </a>
</div>



## Overview

The PROGRESS pipeline consists of three main stages:

1. **Feature Extraction** - Extract visual (DINO) and text (BERT) features from training data
2. **Clustering** - Cluster samples based on combined features and select initial warm-up data
3. **Iterative Training** - Train model iteratively, selecting new samples based on accuracy feedback

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROGRESS Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │ 0. Feature       │    │ 1. Clustering    │    │ 2. Iterative         │   │
│  │    Extraction    │───▶│    & Selection   │───▶│    Training          │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│  • DINO features          • K-means clustering    • Warm-up training        │
│  • BERT features          • Initial 50K/60K       • Accuracy prediction     │
│    (questions)              selection             • Progressive selection   │
│                                                   • AutoDirector loop       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
PROGRESS/
├── 0_Feature_Extraction/     # Extract visual and text features
│   ├── dino_features.py      # DINO image feature extraction
│   ├── questions_features.py # BERT text feature extraction
│   └── run_feature_extraction.sh
│
├── 1_Clustering/             # Cluster data and select initial samples
│   ├── tinyllava/            # Clustering implementation
│   │   └── eval/score/coincide/
│   │       ├── compute_centroids.py    # K-means clustering
│   │       ├── cluster_transferability.py
│   │       └── cluster_wise_prune.py   # Sample selection
│   ├── 0_clustering_dino_bert.sh       # Run clustering
│   └── 2_convert_chosen_index.sh       # Convert indices to JSON
│
├── 2_PROGRESS_Training/      # Iterative training pipeline
│   ├── llava/                # LLaVA model code
│   ├── scripts/              # Training utilities
│   └── Training_Script/      # Training bash scripts
│       ├── run_warmup_pipeline.sh      # Full warm-up pipeline
│       ├── Selection_0_AutoDirector.sh # Automatic training loop
│       └── ...
│
└── utils/                    # Shared utilities
    ├── acc_sort.py           # Accuracy-based sample selection
    ├── concat_jsons.py       # JSON file manipulation
    ├── check_internVL.py     # InternVL accuracy validation
    └── ...
```

## Dataset

### LLaVA-1.5

Follow the visual instruction tuning dataset download guides in the [official LLaVA GitHub page](https://github.com/haotian-liu/LLaVA). Place the downloaded files in `PROGRESS/2_PROGRESS_Training/playground/data`. Also, prepare for the evaluation benchmark datasets by following the instructions on the page.

## Quick Start

### Prerequisites

```bash
# Install clustering dependencies (adapted from [COINCIDE](https://github.com/G-JWLee/COINCIDE_code))
cd PROGRESS/1_Clustering
pip install -e .
pip install -e ".[train]"

# Install training dependencies
cd PROGRESS/2_PROGRESS_Training
pip install -e .
```

Required models:
- Base LLM (e.g., `vicuna-7b-v1.5`)
- Pretrained MM projector (from LLaVA pretraining)

### Step 1: Feature Extraction

Extract visual and text features from your training data:

```bash
cd PROGRESS/0_Feature_Extraction

bash run_feature_extraction.sh \
    --data_path /path/to/llava_v1_5_mix665k.json \
    --image_dir /path/to/images \
    --output_dir /path/to/features
```

**Outputs:**
- `q_features.npy` - BERT embeddings of questions
- `dino_features.npy` - DINOv2 image features

### Step 2: Clustering and Initial Selection

Cluster samples and select initial warm-up data. The clustering code is adapted from [COINCIDE](https://github.com/G-JWLee/COINCIDE_code).

```bash
cd PROGRESS/1_Clustering

# Edit paths in the script first
bash 0_clustering_dino_bert.sh
```

**Outputs (Stage 1 - 1000 centers for annotation):**
- `1000_msa_save_folder_dino-bert/kmeans_centroids.npy` - Cluster centroids
- `1000_msa_save_folder_dino-bert/nearest_cent.npy` - Cluster assignments for each sample
- `1000_msa_save_folder_dino-bert/dist_to_cent.npy` - Distances to centroids

**Outputs (Stage 2 - 5000 centers for selection):**
- `5000_msa_save_folder_dino-bert/kmeans_centroids.npy` - Cluster centroids
- `selected_indices_0.075_dino-bert.npy` - Indices for ~50K samples
- `selected_indices_0.09_dino-bert.npy` - Indices for ~60K samples

Then convert indices to JSON format:

```bash
# Edit paths in the script first
bash 2_convert_chosen_index.sh
```

This script uses 1000 centers for annotation (cluster_id field) and 5000-center indices for selection:

**Outputs:**
- `llava_665k_with_cluster_id.json` - All 665K samples annotated with 1000-center cluster IDs
- `Top_50K.json` - First warm-up data (~50K samples, selected using 5000 centers, annotated with 1000-center IDs)
- `Top_50K_to_60K.json` - Second warm-up data (~10K samples)
- `Top_60K.json` - Combined warm-up data (~60K samples)
- `Remaining_605K.json` - Remaining data for iterative selection

### Step 3: Training

#### Option A: Full Automated Pipeline (Recommended)

Run the complete warm-up pipeline:

```bash
cd PROGRESS/2_PROGRESS_Training/Training_Script

# Edit configuration variables in run_warmup_pipeline.sh first
bash run_warmup_pipeline.sh
```

Then run the iterative training loop:

```bash
# Edit configuration variables in Selection_0_AutoDirector.sh first
bash Selection_0_AutoDirector.sh
```


## Pipeline Details

### Stage 0: Feature Extraction

**Purpose:** Create feature representations that capture both visual and textual content of each sample.

| Script | Description |
|--------|-------------|
| `dino_features.py` | Extracts DINOv2 features from images using ViT-L/14 |
| `questions_features.py` | Extracts BERT embeddings from question text |

**Why both features?**
- **DINO features** capture visual content and image complexity
- **BERT features** capture question semantics and task type
- Combined features enable clustering by both visual and linguistic similarity

### Stage 1: Clustering

**Purpose:** Group similar samples and select diverse, representative initial training data.

> **Note:** The clustering implementation is adapted from [COINCIDE](https://github.com/G-JWLee/COINCIDE_code). Please cite their work if you use this component.

| Script | Description |
|--------|-------------|
| `compute_centroids.py` | Runs K-means clustering on combined DINO+BERT features |
| `cluster_transferability.py` | Computes transferability scores between clusters |
| `cluster_wise_prune.py` | Selects samples from each cluster based on diversity |

**Two-stage clustering approach:**
- **1000 centers** - For annotating all samples with cluster IDs (used for accuracy tracking)
- **5000 centers** - For selecting top 50K/60K warm-up samples (finer granularity for selection)

**Key parameters:**
- `ncentroids_annotation=1000` - Number of clusters for annotation
- `ncentroids_selection=5000` - Number of clusters for selection
- `fraction=0.075` - Fraction of data to select (~50K from 665K)
- `temp=0.1` - Temperature for softmax selection

**Selection strategy:**
1. Cluster all samples using K-means with 1000 centers (for annotation)
2. Cluster all samples using K-means with 5000 centers (for selection)
3. Compute cluster transferability on 5000-center clustering
4. Select samples using 5000-center indices
5. Annotate selected samples with 1000-center cluster IDs

### Stage 2: Iterative Training

**Purpose:** Train the model progressively, selecting new samples based on learning progress.

#### Warm-up Phase

```
50K samples (step 391) → 60K samples (step 469) → Initial selection
```

1. Train on Top 50K samples until step 391
2. Predict accuracy on trained samples
3. Validate with InternVL
4. Continue training to step 469 with next 10K
5. Predict accuracy and validate again
6. Select first batch of new samples based on accuracy change

#### Iterative Selection Phase

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoDirector Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │  Train   │───▶│ Predict  │───▶│ InternVL │───▶│ Select │ │
│  │          │    │ Accuracy │    │  Check   │    │ Samples│ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       ▲                                              │      │
│       └──────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Selection criteria:**
- Samples with **largest accuracy improvement** are prioritized
- Uses softmax selection with temperature for stochasticity
- Selects 7500 samples per iteration by default

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NCENTROIDS_ANNOTATION` | 1000 | Number of K-means clusters for annotation |
| `NCENTROIDS_SELECTION` | 5000 | Number of K-means clusters for selection |
| `SELECTION_NUM` | 7500 | Samples selected per iteration |
| `TEMPERATURE` | 0.5 | Softmax temperature for selection |
| `SAMPLES_PER_CLUSTER` | 100 | Randomly selected samples for accuracy prediction |
| `FINAL_STEP` | 1039 | Final training step |

### Data File Naming Convention

| File Pattern | Description |
|--------------|-------------|
| `Top_50K.json` | Initial 50K warm-up samples |
| `Top_60K.json` | Combined 60K warm-up samples |
| `Remaining_after_{step}.json` | Unselected samples after step |
| `Selected_until_{step}.json` | Accumulated selected samples up to step |
| `fastest_acc_data_{num}.json` | Newly selected samples for training |

## Requirements

### Hardware
- 4+ GPUs recommended for training (A100 or similar)
- GPU memory: 40GB+ for training, 24GB+ for inference

### Software
- Python 3.8+
- PyTorch 2.0+
- transformers
- deepspeed
- faiss-gpu (for clustering)
- lmdeploy (for fast inference)

## Citation

If you use PROGRESS in your research, please cite:

```bibtex
@article{chandhok2025learning,
  title={Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection},
  author={Chandhok, Shivam and Yang, Qian and Manas, Oscar and Jain, Kanishk and Sigal, Leonid and Agrawal, Aishwarya},
  journal={arXiv preprint arXiv:2506.01085},
  year={2025}
}
```