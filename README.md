# Speaker-Listener RL for Summarization (DSC-Capstone-A10-2)

This repo implements a speaker-listener training loop where:
- A **speaker** (causal LM, default GPT-2) generates candidate summaries.
- A **listener** (BERTScore-based scorer) compares candidates against the source text.
- The speaker is optimized with **DPO** (Direct Preference Optimization) from listener preferences.

## Reproducibility Overview

There are two replication paths:

1. **Main path (recommended):** train with `speaker_listener_rl/training/dpo_with_listener_wandb.py` on SimpleWiki passages.
2. **Auxiliary CHILDES path:** generate very short utterance summaries and score pairs.

Most users should follow Path 1.

## Repository Structure

- `speaker_listener_rl/data/` data prep scripts and JSONL loaders
- `speaker_listener_rl/scripts/` baseline generation and pair-building scripts
- `speaker_listener_rl/listener/` BERTScore listener and scoring scripts
- `speaker_listener_rl/training/` DPO training and model testing scripts
- `speaker_listener_rl/outputs/` generated artifacts

## Environment Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (if using Weights & Biases):

```bash
wandb login
```

## Hardware Notes

- GPU is strongly recommended.
- Scripts auto-select `cuda` if available, otherwise CPU.
- CPU runs are possible but slow.

## Path 1: Reproduce DPO Training (SimpleWiki)

### 1) Build SimpleWiki JSONL passages (from provided `.train` file)

```bash
python speaker_listener_rl/data/make_wiki.py
```

Expected output file:
- `speaker_listener_rl/data/simple_wiki_passages.jsonl`

### 2) Run DPO training with listener preferences + W&B logging

```bash
python speaker_listener_rl/training/dpo_with_listener_wandb.py \
  --policy_model gpt2 \
  --reference_model gpt2 \
  --input_path speaker_listener_rl/data/simple_wiki_passages.jsonl \
  --output_path speaker_listener_rl/outputs/dpo_simplewiki \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 1e-5 \
  --alpha 0.01 \
  --alpha_k 2 \
  --beta 0.1 \
  --max_length 256 \
  --top_p 0.9 \
  --temperature 0.7 \
  --max_new_tokens 16 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --score_gap_min 0.0 \
  --max_pair_similarity 0.85 \
  --max_resample_tries 2 \
  --listener_model_type bert-base-uncased \
  --listener_batch_size 8 \
  --test_size 0.1 \
  --split_seed 42 \
  --run_validation \
  --validation_max_examples 128 \
  --wandb_project speaker-listener-dpo \
  --wandb_run_name gpt2-simplewiki-run1
```

Outputs:
- Checkpoints: `speaker_listener_rl/outputs/dpo_simplewiki/checkpoint-*`
- Final model: `speaker_listener_rl/outputs/dpo_simplewiki/final_model`

### 3) Sample from the trained model

```bash
python speaker_listener_rl/training/test_model.py \
  --model_dir speaker_listener_rl/outputs/dpo_simplewiki/final_model \
  --prompt "Keywords only. The quick brown fox jumps over the lazy dog. Summary:" \
  --max_new_tokens 16
```

## Path 2: CHILDES Pipeline (Auxiliary)

### 1) Build CHILDES utterances

```bash
python speaker_listener_rl/data/make_childes.py
```

Produces:
- `speaker_listener_rl/data/childes_utterances.jsonl`

### 2) Generate short speaker outputs (K=2)

```bash
python speaker_listener_rl/scripts/run_speaker_childes.py
```

Produces:
- `speaker_listener_rl/outputs/generations_childes_K2.jsonl`

### 3) Convert generations to candidate-reference pairs

```bash
python speaker_listener_rl/scripts/make_pairs.py
```

Produces:
- `speaker_listener_rl/outputs/childes_pairs_K2.jsonl`

### 4) Score pairs with listener (BERTScore)

```bash
python speaker_listener_rl/listener/scripts/score_pairs.py \
  --infile speaker_listener_rl/outputs/childes_pairs_K2.jsonl \
  --outfile speaker_listener_rl/outputs/childes_pairs_K2_scored.jsonl \
  --model_type roberta-large \
  --batch_size 16
```

Produces:
- `speaker_listener_rl/outputs/childes_pairs_K2_scored.jsonl`

## Optional Baseline Generation (SimpleWiki)

```bash
python speaker_listener_rl/scripts/run_speaker_wiki.py
```

Produces:
- `speaker_listener_rl/outputs/speaker_summaries_simplewiki_K10.jsonl`

## Key Reproducibility Controls

- Set `--split_seed` in DPO training for deterministic train/test split.
- Keep model names and decoding settings fixed (`top_p`, `temperature`, `max_new_tokens`).
- Keep listener model fixed (`--listener_model_type`).
- Save exact CLI command used for each run.

## Common Issues

- **OOM on GPU:** reduce `--batch_size`, `--listener_batch_size`, or `--max_length`.
- **No W&B logging:** ensure `wandb login` is completed and `--wandb_project` is set.
- **Slow runtime:** use GPU and reduce `--validation_max_examples` / epochs for smoke tests.

## Minimal Smoke Test

If you only want to verify the pipeline quickly:

```bash
python speaker_listener_rl/data/make_wiki.py
python speaker_listener_rl/training/dpo_with_listener_wandb.py \
  --policy_model gpt2 \
  --reference_model gpt2 \
  --input_path speaker_listener_rl/data/simple_wiki_passages.jsonl \
  --output_path speaker_listener_rl/outputs/dpo_smoke \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 2 \
  --validation_max_examples 16 \
  --wandb_project speaker-listener-dpo \
  --wandb_run_name smoke-test
```
