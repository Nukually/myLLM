# MyLLM Training

This repository contains the training code for MyLLM, supporting advanced features like Mixture of Experts (MoE), RoPE, and DeepSpeed distributed training.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your secrets (e.g., `SWANLAB_API_KEY`):
   ```bash
   # .env
   SWANLAB_API_KEY=your_api_key_here
   ```

## Distributed Training with DeepSpeed

We support multi-GPU training using DeepSpeed. The configuration file is located at `trainer/ds_config.json`.

### Running Pretraining

To run Pretraining on multiple GPUs (e.g., 2 GPUs), use the following command:

```bash
deepspeed --num_gpus=2 trainer/train_pretrain.py --use_wandb
```

### Running DPO Training

To run Direct Preference Optimization (DPO) training on multiple GPUs (e.g., 2 GPUs), use the following command:

```bash
deepspeed --num_gpus=2 trainer/train_dpo.py --use_wandb
```

**Parameters:**
- `--num_gpus`: Number of GPUs to use.
- `--deepspeed_config`: Path to DeepSpeed config (default: `trainer/ds_config.json`).
- `--use_wandb`: Enable SwanLab experiment tracking.

## SwanLab Integration

This project uses SwanLab for experiment tracking. The API key is automatically loaded from the `.env` file.
