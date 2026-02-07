# MyLLM

This repository contains the training and inference code for MyLLM, supporting advanced features like Mixture of Experts (MoE), RoPE, and DeepSpeed distributed training.



## Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your secrets (e.g., `SWANLAB_API_KEY` for experiment tracking).

## Training

We support multi-GPU training using DeepSpeed. The configuration file is located at `trainer/ds_config.json`.

### Pre-training

To run pre-training on the full dataset (e.g., using 2 GPUs), run the following command from the project root:

```bash
PATH=/root/miniconda3/envs/minimind/bin:$PATH PYTHONPATH=$PWD deepspeed --num_gpus=2 trainer/train_pretrain.py \
    --data_path dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --save_dir out \
    --log_interval 100 \
    --save_interval 1000
```

**Note:** The `trainer/ds_config.json` has been configured for ~2760 training steps (based on the `pretrain_hq.jsonl` dataset size).

## Inference

To evaluate the trained model or chat with it, use the `eval_llm.py` script.

### Chat Mode (Interactive)

```bash
python eval_llm.py --load_from model --save_dir out --weight pretrain --mode 1
```

### Auto Evaluation Mode

```bash
python eval_llm.py --load_from model --save_dir out --weight pretrain --mode 0
```

**Parameters:**
- `--load_from`: Model source (`model` for local loading).
- `--save_dir`: Directory containing the trained weights.
- `--weight`: Weight prefix (e.g., `pretrain`).
- `--mode`: `0` for auto-test, `1` for interactive chat.
