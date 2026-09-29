# FraudOps

Détection de fraude bancaire sur deux jeux complémentaires, évaluée comme en production : validation temporelle, coût métier sous contrainte de capacité, calibration et équité.

Projet en cours de construction.

## Installation

Prérequis : Python 3.14.

```bash
make install   # crée .venv, installe les dépendances et les hooks pre-commit
make lint
make test
```

## Documentation

- [Cadrage métier](docs/cadrage.md) : coûts, capacité d'investigation, métriques retenues.
- [Journal des décisions](docs/DECISIONS.md) : les choix du projet et leur raison.
