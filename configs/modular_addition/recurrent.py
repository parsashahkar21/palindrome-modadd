from dataclasses import dataclass
from frozendict import frozendict
from data.modular_addition import VOCAB_SIZE, PAD_TOKEN, MODULUS

@dataclass(frozen=True)
class RecurrentModelConfig:
    vocab_size: int = VOCAB_SIZE       # size of vocabulary
    pad_token: int = PAD_TOKEN         # padding token index
    embed_dim: int = 128               # embedding dimension
    hidden_dim: int = 256              # hidden state size of GRU
    num_layers: int = 2                # number of stacked GRU layers
    num_classes: int = MODULUS         # output classes (modulus)
    bidirectional: bool = False        # unidirectional GRU
    dropout: float = 0.1               # dropout rate between layers

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64               # batch size for training
    max_epochs: int = 20               # max number of training epochs
    k_folds: int = 5                   # K-fold cross-validation
    optimizer: frozendict = frozendict({
        "lr": 0.001,                   # learning rate
        "weight_decay": 1e-5            # weight decay for regularization
    })

@dataclass(frozen=True)
class Config:
    model: RecurrentModelConfig = RecurrentModelConfig()
    train: TrainingConfig = TrainingConfig()

config = Config()
