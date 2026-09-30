# FraudOps

[![CI](https://github.com/MohamedRadhi52/fraudops/actions/workflows/ci.yml/badge.svg)](https://github.com/MohamedRadhi52/fraudops/actions/workflows/ci.yml)

Détection de fraude carte bancaire, évaluée comme en production : le modèle est ré-entraîné chaque semaine, une fraude n'est connue que 7 jours après la transaction, et l'équipe d'enquête ne peut contrôler que 100 cartes par jour.

![Coût de la fraude sur 8 semaines de test selon la stratégie](reports/figures/cost_by_strategy.png)

| Modèle | Card Precision@100 | AUC-PR | Coût sur 8 semaines |
|---|---|---|---|
| Règles métier | 14,6 % [13,7 ; 15,6] | 0,37 [0,35 ; 0,39] | 85,2 k€ [80,7 ; 89,7] |
| Régression logistique | 19,1 % [18,1 ; 20,1] | 0,61 [0,58 ; 0,63] | 66,3 k€ [61,4 ; 71,3] |
| **LightGBM** | **19,5 %** [18,5 ; 20,5] | **0,66** [0,65 ; 0,68] | **54,0 k€** [50,4 ; 58,0] |
| Modèle parfait | 27,6 % | 1 | 15,4 k€ [14,8 ; 16,2] |

Résultats sur 8 semaines de test, avec des intervalles de confiance à 95 % obtenus par bootstrap sur les 56 jours. La Card Precision@100 est la part de vraies fraudes parmi les 100 cartes contrôlées chaque jour. Le coût additionne 10 € par carte contrôlée et le montant des fraudes non stoppées.

**Ce qu'il faut retenir.**

- **Un coût divisé par 5,5.** Avec un seuil choisi pour minimiser le coût, LightGBM ramène le coût de la fraude de 300 k€ à 54 k€ sur 8 semaines, en contrôlant 35 cartes par jour sur les 100 possibles.
- **Presque tout ce qui échappe est indétectable.** LightGBM stoppe 95 à 100 % des fraudes repérables. L'essentiel de ce qui manque vient d'un terminal compromis dont aucune fraude n'est encore connue : ni le montant ni l'historique du client ne trahissent ces fraudes.
- **Des probabilités fiables.** Contrôler une carte dès que probabilité × montant dépasse 10 € coûte 52 k€, sans aucun seuil à régler. Entraîné avec pondération des classes, le même modèle gonflerait ses probabilités et coûterait 91 k€.
- **LightGBM face à la régression logistique.** Il ne gagne que 0,4 point de Card Precision@100 [0,2 ; 0,7], mais 12 k€ de coût : son avantage se joue sur les cartes les plus suspectes, là où se place le seuil (97,7 % de fraudes parmi les 10 premières chaque jour, contre 93,9 %).

Données : 1,8 M de transactions simulées d'après le Fraud Detection Handbook (ULB, 2022). Outils : pandas, DuckDB, PySpark, scikit-learn, LightGBM, MLflow, pytest, GitHub Actions.

## Méthode

- **Validation préquentielle.** Chaque semaine, le modèle est ré-entraîné sur les 28 derniers jours dont les étiquettes sont connues, puis score la semaine suivante. Les 4 premières semaines servent aux réglages, les 8 suivantes aux résultats.
- **Cartes bloquées.** Une carte dont une fraude est connue est bloquée et ses transactions ne sont plus scorées. Cela retire la moitié des fraudes des semaines de test, celles que la banque connaît déjà sans modèle, et ramène le plafond de la Card Precision@100 à 0,28.
- **Règles métier.** Trois règles tirées de l'exploration : montant supérieur à 220 €, fraude récente déjà connue sur le terminal, montant trois fois supérieur à l'habitude du client. À égalité de règles, les plus gros montants passent en premier.
- **LightGBM.** Arrêt précoce sur la dernière semaine de la fenêtre d'entraînement et 6 combinaisons d'hyperparamètres comparées sur les semaines de validation, toutes suivies dans MLflow (`make mlflow`).

## Seuil et coût

![Coût total selon le nombre de cartes contrôlées par jour](reports/figures/cost.png)

Pour chaque modèle, le seuil d'alerte qui minimise le coût est choisi sur les semaines de validation, puis appliqué tel quel aux semaines de test (`make cost`). Les cartes au-dessus du seuil sont contrôlées, dans la limite de 100 par jour.

| Stratégie | Coût sur 8 semaines | Contrôles par jour |
|---|---|---|
| Aucun contrôle | 299,6 k€ [279,8 ; 320,1] | 0 |
| LightGBM, 100 contrôles chaque jour | 85,9 k€ [82,9 ; 89,2] | 100 |
| LightGBM, seuil de 0,08 | 54,0 k€ [50,4 ; 58,0] | 35 |

**Le seuil que je recommande : 0,08 sur la probabilité donnée par LightGBM.** Il divise le coût par 5,5 par rapport à l'absence de contrôle et économise 32 k€ sur 8 semaines par rapport à l'usage systématique des 100 contrôles : au-dessous de ce seuil, une carte a trop peu de chances d'être frauduleuse pour justifier un contrôle à 10 €. Il n'utilise qu'un tiers de la capacité. Le reste absorbe les pics de fraude et peut servir à des contrôles aléatoires, le seul moyen de mesurer la fraude que le modèle ne voit pas. Deux réserves : ce seuil dépend de l'hypothèse de 10 € par contrôle, et il doit être recalculé à chaque ré-entraînement, car il repose sur la distribution des scores.

Sous 15 contrôles par jour, les règles métier font mieux que LightGBM : elles traitent d'abord les gros montants, alors que LightGBM classe les cartes par probabilité de fraude, sans tenir compte du montant en jeu.

## Calibration

![Courbes de fiabilité, avant et après calibration](reports/figures/calibration.png)

Un score qui alimente une décision doit être une vraie probabilité. C'est ce qui permet de contrôler une carte quand sa perte attendue, probabilité × montant, dépasse le coût d'un contrôle, sans chercher de seuil. La calibration isotonique est ajustée sur les semaines de validation (`make calibration`).

| Variante | Score de Brier | Probabilité moyenne | Coût avec la règle de perte attendue |
|---|---|---|---|
| LightGBM | 0,00296 | 0,66 % | 52,0 k€ [48,5 ; 55,7], 28 contrôles par jour |
| LightGBM pondéré | 0,02174 | 10,90 % | 91,3 k€ [87,7 ; 94,9], 100 contrôles par jour |
| LightGBM pondéré, calibré | 0,00314 | 0,63 % | 53,5 k€ [49,8 ; 57,4], 30 contrôles par jour |

Les semaines de test comptent 0,64 % de fraudes. Entraîné sans pondération des classes, LightGBM est déjà bien calibré : la calibration isotonique ne change presque rien (score de Brier de 0,00297), et la règle de perte attendue coûte 52,0 k€, un peu moins que le seuil optimisé de la section précédente, sans aucun réglage. Elle corrige aussi le défaut vu plus haut à faible volume, puisqu'elle tient compte du montant en jeu. Pondérer les classes, une pratique courante contre le déséquilibre, gonfle les probabilités : la règle contrôle alors 100 cartes par jour et coûte 91 k€. Une calibration ajustée sur la validation ramène ces probabilités au bon niveau (53,5 k€). D'où la règle retenue : pas de pondération ni de SMOTE, et une calibration vérifiée à chaque nouveau modèle.

## Détection

![Cartes frauduleuses parmi les 100 contrôlées chaque jour, par modèle](reports/figures/card_precision.png)

Avec LightGBM, l'équipe trouve en moyenne 19,5 cartes frauduleuses parmi les 100 cartes qu'elle contrôle chaque jour, contre 27,6 pour un modèle parfait. Part des fraudes stoppées, c'est-à-dire dont la carte est contrôlée le jour même ou déjà bloquée :

| Scénario | Fraudes stoppées |
|---|---|
| 1. Montant supérieur à 220 € | 100 % |
| 2. Terminal compromis, fraude déjà connue sur ce terminal | 95 % |
| 2. Terminal compromis, aucune fraude encore connue | 8 % |
| 3. Client compromis | 98 % |

## Ouverture de compte (BAF)

Le second jeu, Bank Account Fraud (Feedzai, NeurIPS 2022), contient 1 million de demandes d'ouverture de compte sur 8 mois, dont environ 1 % de fraudes, avec des attributs sensibles comme l'âge. Sa licence interdit l'usage commercial : le workflow [`baf.yml`](.github/workflows/baf.yml) le télécharge avec un jeton Kaggle gardé en secret, entraîne et évalue les modèles, puis publie seulement les métriques et les figures dans [`reports/baf/`](reports/baf/RESULTS.md). Les données ne sont jamais dans le dépôt.

- **Protocole.** Entraînement sur les mois 0 à 4, arrêt précoce et choix du seuil sur le mois 5, test sur les mois 6 et 7. Le seuil signale 5 % des demandes légitimes du mois de validation, puis reste fixe : le taux de faux positifs obtenu sur le test mesure l'effet de la dérive dans le temps.
- **Comparaison avec le papier.** Le papier fixe le seuil sur le test lui-même. Cette mesure est donnée aussi, pour comparer.
- **Intervalles de confiance.** Bootstrap sur les demandes, apparié entre modèles pour le gain de LightGBM.

## Installation

Prérequis : Python 3.14, et Java 21 pour PySpark (sous Ubuntu : `sudo apt install openjdk-21-jre-headless`).

```bash
make install      # crée .venv, installe les dépendances et les hooks pre-commit
make test         # tests, dont l'égalité des features pandas et Spark
make data         # génère data/transactions.parquet
make explore      # lance les requêtes DuckDB de sql/
make features     # calcule les features avec pandas
make benchmark    # compare pandas et Spark sur le jeu complet
make evaluate     # entraîne et évalue les modèles, écrit reports/ (environ 3 minutes)
make mlflow       # ouvre l'interface MLflow sur http://127.0.0.1:5000
make cost         # choisit le seuil par le coût, écrit reports/
make calibration  # calibre les probabilités, écrit reports/
make report       # evaluate, cost et calibration à la suite
make baf          # BAF, si data/baf/Base.csv a été téléchargé depuis Kaggle
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
