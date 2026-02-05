"""
Reward (preference) model: encoder + linear head that outputs a scalar score
for (prompt, response) pairs. Used in RLHF to prefer chosen over rejected.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class RewardModel(nn.Module):
    """
    Scores (prompt, response) pairs with a single scalar.
    Backbone: Hugging Face encoder (e.g. BERT, DeBERTa); head: linear layer on [CLS].
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            scores: (batch,) scalar per example
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.head(cls).squeeze(-1)

    def score_pairs(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scores_chosen, scores_rejected) for a batch of pairs."""
        score_chosen = self.forward(chosen_ids, chosen_mask)
        score_rejected = self.forward(rejected_ids, rejected_mask)
        return score_chosen, score_rejected
