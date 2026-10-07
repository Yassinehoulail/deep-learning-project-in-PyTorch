"""Fonctions reutilisables pour les experiences PyTorch du projet.

Le notebook principal reste autoportant pour faciliter la lecture pedagogique.
Ce module donne une base propre pour reutiliser certaines fonctions dans un
script Python ou dans des experiences complementaires.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixe les graines aleatoires principales pour rendre les essais reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Retourne cuda si disponible, sinon cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Compte les parametres entrainables d'un modele PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_predictions_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convertit des logits binaires en predictions 0/1."""
    return (torch.sigmoid(logits) >= threshold).long()
