#!/usr/bin/env python3
"""
Evaluate a trained preference model: report accuracy (fraction of pairs
where score(chosen) > score(rejected)).
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import PreferenceDataset
from model import RewardModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF preference model")
    parser.add_argument("--data_path", type=str, default="data/sample_preferences.jsonl")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/reward_model/best_model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    dataset = PreferenceDataset(
        args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Infer base model name from config in checkpoint dir if needed; else use same as training
    model = RewardModel(args.checkpoint_path)
    state_path = os.path.join(args.checkpoint_path, "pytorch_model.bin")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            score_chosen, score_rejected = model.score_pairs(
                chosen_ids, chosen_mask, rejected_ids, rejected_mask
            )
            correct += (score_chosen > score_rejected).sum().item()
            total += chosen_ids.size(0)

    accuracy = correct / total if total else 0.0
    print(f"Preference accuracy: {correct}/{total} = {accuracy:.2%}")


if __name__ == "__main__":
    main()
