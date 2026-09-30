# FraudOps

[![CI](https://github.com/MohamedRadhi52/fraudops/actions/workflows/ci.yml/badge.svg)](https://github.com/MohamedRadhi52/fraudops/actions/workflows/ci.yml)

Détection de fraude carte bancaire, évaluée comme en production : le modèle est ré-entraîné chaque semaine, une fraude n'est connue que 7 jours après la transaction, et l'équipe d'enquête ne peut contrôler que 100 cartes par jour.

![Cartes frauduleuses parmi les 100 contrôlées chaque jour, par modèle](reports/figures/card_precision.png)

| Modèle | Card Precision@100 | AUC-PR |
|---|---|---|
| Règles métier | 14,6 % [13,7 ; 15,6] | 0,37 [0,35 ; 0,39] |
| Régression logistique | 19,1 % [18,1 ; 20,1] | 0,61 [0,58 ; 0,63] |
| **LightGBM** | **19,5 %** [18,5 ; 20,5] | **0,66** [0,65 ; 0,68] |
| Plafond (modèle parfait) | 27,6 % | 1 |

La Card Precision@100 est la part de vraies fraudes parmi les 100 cartes contrôlées chaque jour. Résultats sur 8 semaines de test, avec des intervalles de confiance à 95 % obtenus par bootstrap sur les 56 jours.

**Ce qu'il faut retenir.** LightGBM stoppe 95 à 100 % des fraudes repérables. Il ne manque que les fraudes d'un terminal compromis dont aucune fraude n'est encore connue (8 % stoppées) : tant que les premières étiquettes n'arrivent pas, ni le montant ni l'historique du client ne les trahissent. C'est l'essentiel de l'écart au plafond. Face à la régression logistique, le gain est net en AUC-PR mais faible en Card Precision@100 (+0,4 point [0,2 ; 0,7]) : ici, les features portent l'essentiel du signal.

Données : 1,8 M de transactions simulées d'après le Fraud Detection Handbook (ULB, 2022). Outils : pandas, DuckDB, PySpark, scikit-learn, LightGBM, MLflow, pytest, GitHub Actions.

## Méthode

- **Validation préquentielle.** Chaque semaine, le modèle est ré-entraîné sur les 28 derniers jours dont les étiquettes sont connues, puis score la semaine suivante. Les 4 premières semaines servent aux réglages, les 8 suivantes aux résultats.
- **Cartes bloquées.** Une carte dont une fraude est connue est bloquée et ses transactions ne sont plus scorées. Cela retire la moitié des fraudes des semaines de test, celles que la banque connaît déjà sans modèle, et ramène le plafond de la Card Precision@100 à 0,28.
- **Règles métier.** Trois règles tirées de l'exploration : montant supérieur à 220 €, fraude récente déjà connue sur le terminal, montant trois fois supérieur à l'habitude du client. À égalité de règles, les plus gros montants passent en premier.
- **LightGBM.** Arrêt précoce sur la dernière semaine de la fenêtre d'entraînement et 6 combinaisons d'hyperparamètres comparées sur les semaines de validation, toutes suivies dans MLflow (`make mlflow`).

## Installation

Prérequis : Python 3.14, et Java 21 pour PySpark (sous Ubuntu : `sudo apt install openjdk-21-jre-headless`).

```bash
make install    # crée .venv, installe les dépendances et les hooks pre-commit
make test       # tests, dont l'égalité des features pandas et Spark
make data       # génère data/transactions.parquet
make explore    # lance les requêtes DuckDB de sql/
make features   # calcule les features avec pandas
make benchmark  # compare pandas et Spark sur le jeu complet
make evaluate   # entraîne et évalue les modèles, écrit reports/ (environ 3 minutes)
make mlflow     # ouvre l'interface MLflow sur http://127.0.0.1:5000
```

## Données

### Transactions carte simulées

Le jeu principal est simulé avec les paramètres du *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook* (Le Borgne, Siblini, Lebichot et Bontempi, Université libre de Bruxelles, 2022) : 5 000 clients et 10 000 terminaux placés sur une grille, pendant 183 jours à partir du 1er avril 2018. Chaque client paie sur des terminaux proches de chez lui. `make data` génère le jeu en quelques secondes.

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

## Exploration

Sept requêtes DuckDB, dans [`sql/`](sql), lancées par `make explore`. Voici les constats qui orientent les features.

1. **Le taux de fraude met quatre semaines à se stabiliser.** Il passe de 0,23 % la première semaine à environ 0,9 % à partir de la cinquième, le temps que les compromissions s'accumulent (un terminal reste compromis 28 jours). Les évaluations commenceront après cette montée en charge.
2. **Aucune transaction légitime ne dépasse 220 €.** Une règle sur le montant attrape donc 3 609 fraudes, soit 23,5 %, sans aucune fausse alerte. En revanche, les fraudes des terminaux compromis ont des montants ordinaires (médiane de 46,73 € contre 46,43 € pour les transactions légitimes) : le montant seul ne suffit pas.
3. **Un terminal déjà touché reste risqué.** Quand une fraude y a déjà été confirmée, sur une transaction vieille de 7 à 37 jours (délai d'étiquetage oblige), le taux de fraude atteint 4,48 % contre 0,49 % ailleurs, et ces transactions concentrent 47,5 % des fraudes. D'où les features de risque par terminal, décalées de 7 jours.
4. **Un client compromis dépense plus que d'habitude.** Une fraude du scénario 3 vaut en médiane 3,7 fois la dépense moyenne du client sur les 30 jours précédents, contre 0,98 fois pour une transaction légitime. D'où le nombre de transactions et le montant moyen de chaque client sur 1, 7 et 30 jours.
5. **La Card Precision@100 ne peut pas atteindre 1.** Après le premier mois, on compte en moyenne 78 cartes frauduleuses par jour (entre 55 et 101) sur environ 3 800 cartes actives, et beaucoup ont déjà une fraude connue. Les résultats se lisent donc par rapport au plafond d'un modèle parfait, calculé dans les conditions de l'évaluation.

Enfin, le taux de fraude est le même la nuit et le jour, en semaine et le week-end (entre 0,84 % et 0,89 %) : contrairement au livre, le projet n'utilise pas de feature d'heure ni de jour.

## Features

`make features` calcule 12 features par transaction, en une vingtaine de secondes.

| Famille | Fenêtres | Contenu |
|---|---|---|
| Client | 1, 7 et 30 jours | nombre de transactions et montant moyen, transaction scorée comprise |
| Terminal | 1, 7 et 30 jours, décalées de 7 jours | nombre de transactions et taux de fraude |

Le décalage des features terminal vient du délai d'étiquetage : une fraude n'est confirmée qu'après enquête, environ 7 jours plus tard. Une feature qui utiliserait les étiquettes des 7 derniers jours paraîtrait excellente hors ligne, mais serait impossible à calculer en production. Deux tests le vérifient : les features d'une transaction ne changent pas quand on supprime toutes les transactions postérieures, ni quand on inverse les étiquettes encore inconnues à sa date.

## PySpark

`src/fraudops/spark_features.py` calcule les mêmes features avec des fonctions de fenêtre Spark (`rangeBetween` sur le temps en secondes). Un test vérifie que les deux versions donnent les mêmes valeurs, et `make benchmark` compare leurs temps sur le jeu complet, en vérifiant de nouveau l'égalité sur les 1,8 M de transactions.

| Étape | Temps |
|---|---|
| pandas | 15 s |
| Spark : démarrage de la session | 9 s |
| Spark : calcul et écriture | 32 s |

Mesures faites sur une machine à 1 cœur et 4 Go de mémoire. À ce volume, pandas est plus rapide : Spark paie le démarrage de la JVM et l'organisation de ses tâches, sans rien paralléliser sur un seul cœur. Il devient utile quand les données ne tiennent plus dans la mémoire d'une machine, ou quand plusieurs cœurs ou machines se partagent le travail.

## Documentation

- [Cadrage métier](docs/cadrage.md) : coûts, capacité d'investigation, métriques retenues.
- [Journal des décisions](docs/DECISIONS.md) : les choix du projet et leur raison.
