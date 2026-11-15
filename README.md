# Sequential Deep Learning

Train recurrent and transformer models on two sequence tasks with k-fold cross-validation.

## Tasks

- **Palindrome** — binary classification (is the sequence a palindrome?)
- **Modular addition** — predict the sum of digits modulo M

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
# Palindrome + transformer
python train.py -t pal -m t

# Palindrome + recurrent
python train.py -t pal -m r

# Modular addition + transformer
python train.py -t mod -m t

# Modular addition + recurrent
python train.py -t mod -m r
```

Optional: `-s` / `--seed` (default `42`).

Plots are written to `output/plots/`.

## Layout

```
configs/              # model & training hyperparameters
data/                 # datasets
models/               # TransformerClassifier, RecurrentClassifier
lightning_modules/    # PyTorch Lightning wrappers
train.py              # entrypoint
visualization.py      # learning curves & dumbbell plots
```


## Authors

- [Arian Jafari](https://github.com/Arian-jafari)
- [Parsa Shahkar](https://github.com/parsashahkar21)
- [Paniz Roozegar](https://github.com/paniizz)

