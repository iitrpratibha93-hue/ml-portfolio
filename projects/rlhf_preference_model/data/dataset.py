"""
Dataset for preference pairs: (prompt, chosen, rejected).
Returns tokenized (chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask).
"""

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_preferences(path: str) -> List[dict]:
    """Load JSONL with keys: prompt, chosen, rejected."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class PreferenceDataset(Dataset):
    """
    Dataset of (prompt, chosen, rejected) triples.
    Each sample is tokenized as (prompt + chosen) and (prompt + rejected).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        sep: str = " [SEP] ",
    ):
        self.items = load_preferences(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep = sep

    def __len__(self) -> int:
        return len(self.items)

    def _format_pair(self, prompt: str, response: str) -> str:
        return (prompt + self.sep + response).strip()

    def __getitem__(self, idx: int) -> dict:
        row = self.items[idx]
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        chosen_text = self._format_pair(prompt, chosen)
        rejected_text = self._format_pair(prompt, rejected)

        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }
