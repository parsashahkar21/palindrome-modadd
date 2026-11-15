from dataclasses import dataclass
from frozendict import frozendict
from data.palindrome import VOCAB_SIZE, PAD_TOKEN

@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int = VOCAB_SIZE          # in data/palindrome.py => 10
    pad_token: int = PAD_TOKEN            # in data/palindrome.py => 10

    # Model hyperparameters
    embed_dim: int = 128                  # must be divisible by num_heads
    mlp_dim: int = 512                    # usually 4x embed_dim is a good default
    num_heads: int = 8
    num_layers: int = 4
    num_classes: int = 2                 # palindrome is binary classification

    # Positional encoding length
    # dev sequences up to 100, test up to 500 => set >= 500
    max_len: int = 512

    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 10
    k_folds: int = 5
    optimizer: frozendict = frozendict({
        "lr": 3e-4,
        "weight_decay": 1e-2
    })

@dataclass(frozen=True)
class Config:
    model: TransformerModelConfig = TransformerModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()