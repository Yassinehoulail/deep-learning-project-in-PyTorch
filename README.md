# Projet Deep Learning - EMSI Casablanca

Ce projet contient un notebook Jupyter complet en francais pour le sujet :

**Conception, implementation, comparaison et analyse critique de modeles de deep learning pour donnees tabulaires, images et sequences.**

## Structure

- `notebooks/00_projet_deep_learning_emsi.ipynb` : notebook principal.
- `requirements.txt` : dependances Python.
- `models/` : sauvegardes de modeles generees pendant l'execution.
- `figures/` : dossier reserve aux figures exportees.
- `data/` : donnees telechargees automatiquement, notamment Fashion-MNIST.

## Installation

```bash
pip install -r requirements.txt
```

## Execution

Lancer Jupyter puis ouvrir le notebook principal :

```bash
jupyter notebook notebooks/00_projet_deep_learning_emsi.ipynb
```

Le notebook utilise automatiquement `cuda` si un GPU compatible est disponible, sinon `cpu`.
"# deep_learning" 
