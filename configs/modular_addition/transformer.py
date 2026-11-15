from dataclasses import dataclass
from frozendict import frozendict

from data.modular_addition import MODULUS, PAD_TOKEN


@dataclass(frozen=True)
class TransformerModelConfig:
    # vocabulary tokens: 0..MODULUS-1  (مثلاً 0..30)
    # padding token معمولاً == MODULUS (مثلاً 31)
    vocab_size: int = MODULUS
    pad_token: int = PAD_TOKEN

    embed_dim: int = 128
    mlp_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4

    # خروجی modular addition یکی از 0..MODULUS-1 است
    num_classes: int = MODULUS

    # طبق دیتاست/پروژه طول توالی‌های تست تا 40 است
    max_len: int = 40

    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 10
    k_folds: int = 5
    optimizer: frozendict = frozendict({
        "lr": 3e-4,
        "weight_decay": 1e-2,
    })


@dataclass(frozen=True)
class Config:
    model: TransformerModelConfig = TransformerModelConfig()
    train: TrainingConfig = TrainingConfig()


config = Config()