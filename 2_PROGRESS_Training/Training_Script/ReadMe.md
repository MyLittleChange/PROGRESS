# PROGRESS Training Scripts

This directory contains scripts for running the PROGRESS iterative training pipeline.

## Prerequisites

1. Install dependencies:
   ```bash
   cd PROGRESS/2_PROGRESS_Training
   pip install -e .
   ```

2. Prepare required data files:
   - `Top_50K.json` - First 50K samples for warm-up stage 1
   - `Top_50K_to_60K.json` - Next 10K samples for warm-up stage 2
   - `Top_60K.json` - Combined 60K samples (for accuracy prediction)
   - `Remaining_605K.json` - Leftover samples after warm-up

3. Prepare required model files:
   - Base model (e.g., `vicuna-7b-v1.5`)
   - Pretrained MM projector (`.bin` file from pretraining)

## Quick Start: Full Pipeline

### Option 1: Run Unified Warm-up Pipeline (Recommended)

The unified pipeline runs all warm-up stages in sequence:

```bash
# Edit configuration variables in run_warmup_pipeline.sh first
bash run_warmup_pipeline.sh
```

This script automatically:
1. Trains on 50K samples (stops at step 391)
2. Predicts accuracy at step 391
3. Checks with InternVL at step 391
4. Continues training to 60K samples (stops at step 469)
5. Predicts accuracy at step 469
6. Checks with InternVL at step 469
7. Runs initial data selection


## Iterative Selection Training

After warm-up, use AutoDirector to automatically run the iterative training-selection loop:

```bash
# Edit configuration variables in Selection_0_AutoDirector.sh first
bash Selection_0_AutoDirector.sh
```


## Script Descriptions

### Warm-up Scripts

| Script | Description |
|--------|-------------|
| `run_warmup_pipeline.sh` | **Unified pipeline**: Runs all warm-up stages (training + accuracy + InternVL + selection) |
| `0_Warm_up_training.sh` | Two-stage warm-up training only (50K -> 60K) |
| `391_2_predict_accuracy.sh` | Predict accuracy at step 391 (after 50K training) |
| `391_3_check_internVL.sh` | Validate accuracy with InternVL at step 391 |
| `469_2_predict_accuracy.sh` | Predict accuracy at step 469 (after 60K training) |
| `469_3_check_internVL.sh` | Validate accuracy with InternVL at step 469 |

### Selection Scripts

| Script | Description |
|--------|-------------|
| `Selection_0_AutoDirector.sh` | **Automatic loop controller**: Orchestrates the entire iterative training-selection pipeline |
| `Selection_1_LLaVA_Training.sh` | Trains model on selected samples until next selection step |
| `Selection_2_predict_accuracy.sh` | Predicts accuracy on representative samples (includes LoRA merging) |
| `Selection_3_check_internVL.sh` | Validates accuracy predictions using InternVL |
| `Selection_5_predict_conf.sh` | (Optional) Predicts confidence scores (loss) for samples |

## Configuration

All scripts have configurable variables at the top. Key variables to set:

### Model Paths
- `MODEL_PATH` / `MODEL_BASE`: Path to base LLM (e.g., vicuna-7b-v1.5)
- `PRETRAIN_MM_ADAPTER`: Path to pretrained multimodal projector (.bin file)

### Data Paths
- `IMAGE_FOLDER`: Path to image directory
- `DATA_DIR`: Directory containing JSON data files

### Output Paths
- `OUTPUT_DIR` / `BASE_PATH`: Directory for checkpoints and results

### Training Parameters
- `NUM_GPUS`: Number of GPUs for training (default: 4)
- `SELECTION_NUM`: Samples selected per iteration (default: 7500)
- `TEMPERATURE`: Softmax temperature for selection (default: 0.5)
- `SAMPLES_PER_CLUSTER`: Samples per cluster for representative selection (default: 80)
- `FINAL_STEP`: Final training step (default: 1039)

## Pipeline Details

### AutoDirector Loop Logic

The AutoDirector (`Selection_0_AutoDirector.sh`) handles three cases:

1. **Case 1**: Selection results exist -> Run data selection to update step list
2. **Case 2**: Checkpoint exists -> Check accuracy/InternVL status
   - 2a: InternVL results exist -> Copy to main path
   - 2b: Accuracy prediction exists -> Run InternVL check
   - 2c: No accuracy prediction -> Run accuracy prediction
3. **Case 3**: No checkpoint -> Prepare data and train
   - 3a: Representative selection exists -> Run training
   - 3b: No representative selection -> Run data preparation

### Data File Naming Convention

- `Remaining_after_{step}.json`: Leftover samples after selection at {step}
- `Selected_until_{step}.json`: Accumulated selected samples up to {step}
- `Selected_until_{step}_{samples_per_cluster}.json`: Representative samples for accuracy prediction

### Step Tracking

The pipeline uses `step_list.txt` to track training progress:
- Contains training step numbers, one per line
- AutoDirector reads the last 2-3 lines to determine current state
- New steps are appended after each selection round

## Troubleshooting

1. **Missing step_list.txt**: Create initial file with warm-up steps (391, 469)
2. **LoRA merging fails**: Check that base model path is correct
3. **CUDA out of memory**: Reduce `perplexity_batch_size` in accuracy prediction scripts
