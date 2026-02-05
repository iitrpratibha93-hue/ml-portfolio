"""Configuration for RLHF preference model training and evaluation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """Training hyperparameters and paths."""

    data_path: str = "data/sample_preferences.jsonl"
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "checkpoints/reward_model"
    max_length: int = 256
    batch_size: int = 8
    epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    eval_steps: Optional[int] = 100
    save_steps: Optional[int] = 100
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation config."""

    data_path: str = "data/sample_preferences.jsonl"
    checkpoint_path: str = "checkpoints/reward_model/best_model"
    model_name: Optional[str] = None  # inferred from checkpoint if not set
    max_length: int = 256
    batch_size: int = 8
    seed: int = 42
