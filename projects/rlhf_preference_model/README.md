# RLHF Preference (Reward) Model

A production-style implementation of a **preference model** for Reinforcement Learning from Human Feedback (RLHF). The model learns to score (prompt, response) pairs so that preferred (chosen) responses receive higher scores than rejected ones, and can be used as a reward model for policy optimization or for evaluating LLM outputs.

## Features

- Train on preference triples: `(prompt, chosen_response, rejected_response)`
- Loss: pairwise cross-entropy (Bradley-Terry style): maximize `log σ(r_chosen - r_rejected)`
- Hugging Face `transformers` + PyTorch; configurable base encoder (e.g. BERT, DeBERTa)
- Checkpointing, evaluation script, and sample data included

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

## Data format

JSONL with one object per line:

```json
{"prompt": "What is 2 + 2?", "chosen": "2 + 2 equals 4.", "rejected": "2 + 2 is 3."}
```

Sample data: `data/sample_preferences.jsonl`

## Train

```bash
cd projects/rlhf_preference_model
python train_reward_model.py \
  --data_path data/sample_preferences.jsonl \
  --model_name distilbert-base-uncased \
  --output_dir checkpoints/reward_model \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-5
```

## Evaluate

```bash
python evaluate.py \
  --data_path data/sample_preferences.jsonl \
  --checkpoint_path checkpoints/reward_model/best_model
```

Reports **preference accuracy**: fraction of pairs where `score(chosen) > score(rejected)`.

## Project structure

```
rlhf_preference_model/
  README.md           # This file
  config.py           # Training/eval config
  model.py            # RewardModel (encoder + scalar head)
  train_reward_model.py
  evaluate.py
  data/
    dataset.py        # PreferenceDataset
    sample_preferences.jsonl
```

## Use case

Suitable as a reward model in RLHF pipelines: train on human (or AI) preference data, then use the model to score candidate responses for policy gradient or best-of-N selection.
