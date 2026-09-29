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

## Données

### Transactions carte simulées

Le jeu principal est simulé avec les paramètres du *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook* (Le Borgne, Siblini, Lebichot et Bontempi, Université libre de Bruxelles, 2022) : 5 000 clients et 10 000 terminaux placés sur une grille, pendant 183 jours à partir du 1er avril 2018. Chaque client paie sur des terminaux proches de chez lui.

```bash
make data   # écrit data/transactions.parquet en quelques secondes
```

Trois scénarios de fraude, repris du livre :

1. toute transaction de plus de 220 € est frauduleuse ;
2. chaque jour, 2 terminaux sont compromis pendant 28 jours : toutes leurs transactions sont frauduleuses ;
3. chaque jour, les données de 3 clients sont volées pendant 14 jours : en moyenne un tiers de leurs transactions sont frauduleuses, avec un montant multiplié par 5.

| | Ce dépôt | Livre |
|---|---|---|
| Transactions | 1 793 343 | 1 754 155 |
| Fraudes | 15 340 (0,86 %) | 14 681 (0,84 %) |
| Fraudes des scénarios 1, 2 et 3 | 1 134, 9 286 et 4 920 | 978, 9 099 et 4 604 |

Les chiffres diffèrent légèrement de ceux du livre, car le code est une implémentation indépendante avec son propre générateur aléatoire (le code du livre est sous licence GPL-3.0). La graine est fixée : le jeu est identique à chaque génération.

Limite : les données sont simulées et les fraudes suivent des règles connues. Les performances obtenues ici ne se transposent pas telles quelles à des données réelles.

## Documentation

- [Cadrage métier](docs/cadrage.md) : coûts, capacité d'investigation, métriques retenues.
- [Journal des décisions](docs/DECISIONS.md) : les choix du projet et leur raison.
