#!/usr/bin/env python3
"""
Train a preference (reward) model on (prompt, chosen, rejected) triples.
Loss: -log σ(r_chosen - r_rejected). Saves best checkpoint by eval loss.
"""

import argparse
import logging
import os
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import TrainConfig
from data import PreferenceDataset
from model import RewardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def preference_loss(score_chosen: torch.Tensor, score_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry / pairwise cross-entropy: -log σ(chosen - rejected)."""
    return -torch.nn.functional.logsigmoid(score_chosen - score_rejected).mean()


def main():
    parser = argparse.ArgumentParser(description="Train RLHF preference model")
    parser.add_argument("--data_path", type=str, default="data/sample_preferences.jsonl")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="checkpoints/reward_model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = PreferenceDataset(
        args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RewardModel(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            score_chosen, score_rejected = model.score_pairs(
                chosen_ids, chosen_mask, rejected_ids, rejected_mask
            )
            loss = preference_loss(score_chosen, score_rejected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} — train loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            out_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(out_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(out_path)
            logger.info(f"Saved best model to {out_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
