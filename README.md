# FraudOps

[![CI](https://github.com/MohamedRadhi52/fraudops/actions/workflows/ci.yml/badge.svg)](https://github.com/MohamedRadhi52/fraudops/actions/workflows/ci.yml)

Projet de détection de fraude bancaire sur deux jeux de données : des transactions carte simulées d'après le Fraud Detection Handbook (ULB) et des demandes d'ouverture de compte (jeu BAF de Feedzai, NeurIPS 2022). J'ai voulu l'évaluer dans des conditions proches de la réalité d'une banque : le modèle est ré-entraîné chaque semaine, une fraude n'est connue que 7 jours après la transaction, et l'équipe ne peut contrôler que 100 cartes par jour.

Le rapport complet est en ligne : [mohamedradhi52.github.io/fraudops](https://mohamedradhi52.github.io/fraudops/).

![Coût de la fraude sur 8 semaines de test selon la stratégie](reports/figures/cost_by_strategy.png)

| Fraude carte | Card Precision@100 | AUC-PR | Coût sur 8 semaines |
|---|---|---|---|
| Règles métier | 14,6 % [13,7 ; 15,6] | 0,37 [0,35 ; 0,39] | 85,2 k€ [80,7 ; 89,7] |
| Régression logistique | 19,1 % [18,1 ; 20,1] | 0,61 [0,58 ; 0,63] | 66,3 k€ [61,4 ; 71,3] |
| **LightGBM** | **19,5 %** [18,5 ; 20,5] | **0,66** [0,65 ; 0,68] | **54,0 k€** [50,4 ; 58,0] |
| Modèle parfait | 27,6 % | 1 | 15,4 k€ [14,8 ; 16,2] |

| Ouverture de compte (BAF) | Rappel, seuil fixé en validation | Ratio de faux positifs selon l'âge |
|---|---|---|
| LightGBM | 57,4 % [55,5 ; 59,2] | 0,32 [0,31 ; 0,33] |
| LightGBM, un seuil par groupe d'âge | 55,5 % [53,6 ; 57,3] | 0,94 [0,90 ; 0,98] |
| LightGBM sans la variable d'âge | 56,7 % [54,9 ; 58,7] | 0,44 [0,42 ; 0,45] |

Les intervalles de confiance à 95 % viennent d'un bootstrap. La Card Precision@100 est la part de vraies fraudes parmi les 100 cartes contrôlées chaque jour. Le coût compte 10 € par carte contrôlée, plus le montant des fraudes qui passent. Un ratio de faux positifs de 1 veut dire que les plus de 50 ans et les autres sont signalés à tort aussi souvent.

## En bref

- Avec le seuil qui minimise le coût, LightGBM fait passer le coût de la fraude carte de 300 k€ à 54 k€ sur les 8 semaines de test, en contrôlant 35 cartes par jour sur les 100 possibles.
- Il arrête 95 à 100 % des fraudes qu'on peut repérer. Celles qui passent viennent surtout de terminaux compromis dont aucune fraude n'est encore connue, et pour celles-là il n'y a rien dans les données.
- Ses probabilités sont bien calibrées, donc on peut contrôler une carte dès que probabilité × montant dépasse 10 €, sans chercher de seuil (52 k€). Avec une pondération des classes, le même modèle coûterait 91 k€.
- Sur BAF, je retrouve le résultat du papier : avec un seuil unique, les plus de 50 ans sont signalés à tort trois fois plus souvent que les autres (ratio de 0,32). Un seuil par groupe d'âge remonte le ratio à 0,94 et coûte 1,9 point de rappel. Enlever l'âge du modèle ne suffit pas (0,44), parce que d'autres variables portent la même information.

## Méthode

- Validation préquentielle. Chaque semaine, le modèle est entraîné sur les 28 derniers jours dont on connaît les étiquettes, puis il score la semaine suivante. 4 semaines de validation servent aux réglages et 8 semaines de test aux résultats.
- Pas de fuite. Les features du terminal ignorent les 7 derniers jours, dont les étiquettes ne sont pas encore connues, et deux tests le vérifient. Les cartes dont une fraude est déjà connue sont bloquées, donc le modèle n'est jugé que sur les fraudes que la banque ne connaît pas encore.
- Le seuil minimise le coût avec au plus 100 contrôles par jour, et la calibration est vérifiée avec une courbe de fiabilité.
- BAF est sous licence non commerciale, donc tout se passe dans GitHub Actions : le workflow télécharge les données, entraîne, évalue et publie seulement les résultats.

Outils : pandas, DuckDB, PySpark, scikit-learn, LightGBM, MLflow, Fairlearn, FastAPI, Docker, pytest, GitHub Actions.

## Détails

<details>
<summary><b>Seuil et coût</b></summary>

![Coût total selon le nombre de cartes contrôlées par jour](reports/figures/cost.png)

Pour chaque modèle, je choisis le seuil qui minimise le coût sur les semaines de validation, puis je le garde tel quel sur le test (`make cost`). Les cartes au-dessus du seuil sont contrôlées, 100 par jour au maximum.

| Stratégie | Coût sur 8 semaines | Contrôles par jour |
|---|---|---|
| Aucun contrôle | 299,6 k€ [279,8 ; 320,1] | 0 |
| LightGBM, 100 contrôles chaque jour | 85,9 k€ [82,9 ; 89,2] | 100 |
| LightGBM, seuil de 0,08 | 54,0 k€ [50,4 ; 58,0] | 35 |

Je recommanderais le seuil de 0,08 sur la probabilité de LightGBM. Il divise le coût par 5,5 par rapport à aucun contrôle, et il coûte 32 k€ de moins sur 8 semaines que d'utiliser les 100 contrôles tous les jours, parce qu'en dessous de ce seuil une carte a trop peu de chances d'être frauduleuse pour valoir un contrôle à 10 €. Il n'utilise qu'un tiers de la capacité, ce qui laisse de la marge pour les pics et pour des contrôles aléatoires (c'est le seul moyen de mesurer la fraude que le modèle ne voit pas). Par contre, ce seuil dépend de l'hypothèse des 10 € par contrôle, et il faut le recalculer à chaque ré-entraînement.

En dessous de 15 contrôles par jour, les règles métier font mieux que LightGBM, parce qu'elles prennent d'abord les gros montants alors que LightGBM classe seulement par probabilité.

</details>

<details>
<summary><b>Calibration</b></summary>

![Courbes de fiabilité, avant et après calibration](reports/figures/calibration.png)

Avec des probabilités bien calibrées, on peut décider sans chercher de seuil : on contrôle une carte quand sa perte attendue (probabilité × montant) dépasse le coût d'un contrôle. La calibration isotonique est ajustée sur les semaines de validation (`make calibration`).

| Variante | Score de Brier | Probabilité moyenne | Coût avec la règle de perte attendue |
|---|---|---|---|
| LightGBM | 0,00296 | 0,66 % | 52,0 k€ [48,5 ; 55,7], 28 contrôles par jour |
| LightGBM pondéré | 0,02174 | 10,90 % | 91,3 k€ [87,7 ; 94,9], 100 contrôles par jour |
| LightGBM pondéré, calibré | 0,00314 | 0,63 % | 53,5 k€ [49,8 ; 57,4], 30 contrôles par jour |

Il y a 0,64 % de fraudes dans les semaines de test. Entraîné sans pondération, LightGBM est déjà bien calibré : la calibration isotonique ne change presque rien (Brier de 0,00297), et la règle de perte attendue coûte 52,0 k€ sans rien régler. Elle corrige aussi le problème des petits volumes vu plus haut, puisqu'elle tient compte du montant. Si on pondère les classes, ce qu'on fait souvent contre le déséquilibre, les probabilités sont gonflées, la règle contrôle 100 cartes par jour et coûte 91 k€. Une calibration sur la validation corrige ça (53,5 k€). Je n'utilise donc ni pondération ni SMOTE, et je vérifie la calibration à chaque nouveau modèle.

</details>

<details>
<summary><b>Ce qui est stoppé et ce qui passe</b></summary>

![Cartes frauduleuses parmi les 100 contrôlées chaque jour, par modèle](reports/figures/card_precision.png)

Avec LightGBM, l'équipe trouve en moyenne 19,5 cartes frauduleuses parmi les 100 qu'elle contrôle chaque jour, contre 27,6 pour un modèle parfait. Part des fraudes stoppées (la carte est contrôlée le jour même ou déjà bloquée) :

| Scénario | Fraudes stoppées |
|---|---|
| 1. Montant supérieur à 220 € | 100 % |
| 2. Terminal compromis, fraude déjà connue sur ce terminal | 95 % |
| 2. Terminal compromis, aucune fraude encore connue | 8 % |
| 3. Client compromis | 98 % |

Face à la régression logistique, LightGBM gagne peu en Card Precision@100 (+0,4 point [0,2 ; 0,7]) mais 12 k€ de coût. La différence se fait sur les cartes les plus suspectes, là où se place le seuil : 97,7 % de fraudes parmi les 10 premières chaque jour, contre 93,9 %.

</details>

<details>
<summary><b>Ce que le modèle regarde (SHAP)</b></summary>

![Contribution moyenne de chaque variable au score de LightGBM](reports/figures/shap.png)

Valeurs SHAP du modèle de production (`make explain`), c'est-à-dire LightGBM entraîné sur les 28 derniers jours dont les étiquettes sont connues, puis expliqué sur la semaine suivante avec le TreeSHAP intégré à LightGBM.

Sur l'ensemble des transactions, le modèle regarde surtout l'habitude de dépense du client (montant moyen sur 7 et 30 jours). Sur les 599 transactions qu'il signale (72 % de fraudes), ce sont le montant et le taux de fraude du terminal sur 7 jours qui comptent le plus.

Par exemple, pour une transaction de 913,60 € d'un client qui dépense en moyenne 216 € sur 30 jours, la perte attendue est de 851 €, donc la carte est contrôlée. Les trois raisons sont le montant, puis la dépense moyenne du client sur 7 jours et sur 1 jour, déjà gonflées par 7 autres fraudes de la semaine pas encore connues. L'API renvoie ces trois raisons avec chaque score.

Attention, SHAP explique le modèle et pas la fraude. Des variables corrélées se partagent le crédit un peu au hasard (les taux de fraude du terminal sur 1, 7 et 30 jours se recoupent), et une variable peut en remplacer une autre, comme l'âge sur BAF. Une explication qui ne cite pas une variable sensible ne prouve donc pas que le modèle ne l'utilise pas.

</details>

<details>
<summary><b>API de scoring et Docker</b></summary>

L'API reçoit les features d'une transaction (calculées en amont) et renvoie la probabilité de fraude, la perte attendue, la décision et les trois raisons principales du score. `make api` la lance en local et `make docker` dans son conteneur. La CI construit l'image et la teste, et la documentation interactive est sur `/docs`.

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d @tests/transaction.json
```

```json
{
  "probability": 0.932,
  "expected_loss": 851.49,
  "decision": "contrôler",
  "reasons": [
    {"feature": "tx_amount", "label": "montant de la transaction", "contribution": 6.71},
    {"feature": "customer_avg_amount_7d", "label": "montant moyen du client sur 7 jours", "contribution": 2.08},
    {"feature": "customer_avg_amount_1d", "label": "montant moyen du client sur 1 jour", "contribution": 0.54}
  ]
}
```

La carte est contrôlée quand la perte attendue (probabilité × montant) dépasse le coût d'un contrôle, soit 10 €. Les contributions sont des valeurs SHAP en log-odds.

</details>

<details>
<summary><b>Ouverture de compte (BAF) et équité selon l'âge</b></summary>

Le jeu Bank Account Fraud (Feedzai, NeurIPS 2022) contient 1 million de demandes d'ouverture de compte sur 8 mois, avec 1 à 1,4 % de fraudes selon la période et des attributs sensibles comme l'âge. Sa licence interdit l'usage commercial. Le workflow [`baf.yml`](.github/workflows/baf.yml) le télécharge avec un jeton Kaggle gardé en secret, entraîne et évalue les modèles, puis publie seulement les métriques et les figures dans [`reports/baf/`](reports/baf/RESULTS.md). Les données ne sont jamais dans le dépôt.

J'entraîne sur les mois 0 à 4, je fais l'arrêt précoce et je fixe le seuil (5 % de faux positifs) sur le mois 5, et je teste sur les mois 6 et 7. Mesuré comme dans le papier, avec le seuil fixé sur le test lui-même, LightGBM a 54,0 % de rappel [52,1 ; 55,8], contre 49,6 % [47,7 ; 51,5] pour la régression logistique. Avec le seuil fixé sur le mois de validation, il signale en fait 5,9 % des demandes légitimes du test au lieu de 5 %, parce que les scores dérivent d'un mois à l'autre.

![Compromis entre rappel et égalité des faux positifs selon l'âge](reports/baf/figures/tradeoff.png)

Avec un seuil unique, 13,7 % des demandes légitimes des 50 ans et plus sont signalées, contre 4,4 % pour les autres. Ça fait un ratio de 0,32, ce que le papier trouve pour ses meilleurs modèles. Avec un seuil par groupe, réglé à 5 % de faux positifs sur la validation pour chacun, les deux groupes sont presque à égalité (6,4 % et 6,0 %) et on perd 1,9 point de rappel [0,7 ; 3,1]. C'est le même principe que le `ThresholdOptimizer` de Fairlearn, mais appliqué directement au point de fonctionnement du protocole, que cet outil ne permet pas d'imposer.

Ça ne règle pas tout. Les 40 à 49 ans passent de 6,8 % à 9,1 % de faux positifs, parce que l'égalité entre deux groupes n'efface pas les écarts entre tranches d'âge. Et retirer l'âge ne remonte le ratio qu'à 0,44, car d'autres variables portent la même information.

![Demandes légitimes signalées par tranche d'âge](reports/baf/figures/fpr_by_age.png)

Ce n'est pas au data scientist de choisir le compromis entre rappel et égalité des faux positifs : il le mesure, et la décision revient au métier, à la conformité et au juridique. De toute façon, quand les taux de fraude diffèrent selon l'âge, on ne peut pas égaliser en même temps les faux positifs, les faux négatifs et la calibration. Le règlement européen sur l'IA exclut la détection de fraude financière des systèmes à haut risque, mais le droit de la non-discrimination et le RGPD s'appliquent quand même.

</details>

<details>
<summary><b>Dérive (PSI)</b></summary>

Le PSI (population stability index) compare la distribution d'une variable à celle d'une période de référence. Les seuils d'alerte que j'utilise :

| PSI | Lecture | Action |
|---|---|---|
| moins de 0,1 | stable | rien |
| de 0,1 à 0,25 | dérive modérée | surveiller le taux d'alertes et la calibration, recalibrer si besoin |
| plus de 0,25 | dérive forte | ré-entraîner le modèle et recalculer le seuil |

Sur le jeu carte, semaine par semaine (`make drift`), aucune variable ni le score ne dépassent 0,01 : le simulateur ne bouge pas une fois la montée en charge passée, c'est une limite des données simulées. Le score doit être suivi sur un modèle figé. Si on compare les scores des modèles ré-entraînés chaque semaine, on trouve des PSI jusqu'à 0,7 alors que les données ne changent pas, juste parce que chaque modèle a sa propre échelle de scores.

Sur BAF, mois par mois par rapport aux mois d'entraînement, la dérive est nette. Six variables dépassent 0,25 sur les deux mois de test, surtout les vitesses de demandes : `velocity_4w` monte à 3,7 puis 4,0, et `velocity_24h` à 1,6 puis 2,6. Le PSI du score reste pourtant sous 0,01, alors que le taux de faux positifs passe de 5 % à 5,9 % au seuil fixé en validation. Le PSI calculé par déciles voit mal ce qui se passe dans la queue de la distribution, là où se trouve le seuil, donc il faut aussi suivre directement le taux d'alertes.

![PSI mensuel des variables de BAF les plus instables](reports/baf/figures/psi_monthly.png)

</details>

<details>
<summary><b>Données simulées</b></summary>

Le jeu principal est simulé avec les paramètres du *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook* (Le Borgne, Siblini, Lebichot et Bontempi, Université libre de Bruxelles, 2022) : 5 000 clients et 10 000 terminaux placés sur une grille, pendant 183 jours à partir du 1er avril 2018. Chaque client paie sur des terminaux proches de chez lui. `make data` génère le jeu en quelques secondes.

Les trois scénarios de fraude viennent du livre :

1. toute transaction de plus de 220 € est frauduleuse ;
2. chaque jour, 2 terminaux sont compromis pendant 28 jours et toutes leurs transactions sont frauduleuses ;
3. chaque jour, les données de 3 clients sont volées pendant 14 jours, et en moyenne un tiers de leurs transactions sont frauduleuses, avec un montant multiplié par 5.

| | Ce dépôt | Livre |
|---|---|---|
| Transactions | 1 793 343 | 1 754 155 |
| Fraudes | 15 340 (0,86 %) | 14 681 (0,84 %) |
| Fraudes des scénarios 1, 2 et 3 | 1 134, 9 286 et 4 920 | 978, 9 099 et 4 604 |

Les chiffres sont un peu différents de ceux du livre, parce que j'ai réécrit le simulateur (le code du livre est sous licence GPL-3.0) et qu'il a son propre générateur aléatoire. La graine est fixée, donc le jeu est identique à chaque génération.

</details>

<details>
<summary><b>Exploration SQL</b></summary>

Sept requêtes DuckDB dans [`sql/`](sql), lancées par `make explore`. Voici ce qu'elles montrent et pourquoi ça compte pour les features.

1. Le taux de fraude met quatre semaines à se stabiliser. Il passe de 0,23 % la première semaine à environ 0,9 % à partir de la cinquième, le temps que les compromissions s'accumulent (un terminal reste compromis 28 jours). Les évaluations commencent donc après cette période.
2. Aucune transaction légitime ne dépasse 220 €. Une règle sur le montant attrape donc 3 609 fraudes (23,5 %) sans fausse alerte. Par contre, les fraudes des terminaux compromis ont des montants normaux (médiane de 46,73 € contre 46,43 € pour les légitimes), donc le montant seul ne suffit pas.
3. Un terminal déjà touché reste risqué. Quand une fraude y a déjà été confirmée sur une transaction vieille de 7 à 37 jours (à cause du délai d'étiquetage), le taux de fraude monte à 4,48 % contre 0,49 % ailleurs, et ces transactions regroupent 47,5 % des fraudes. D'où les features de risque par terminal, décalées de 7 jours.
4. Un client compromis dépense plus que d'habitude. Une fraude du scénario 3 vaut en médiane 3,7 fois sa dépense moyenne des 30 jours précédents, contre 0,98 fois pour une transaction légitime. D'où le nombre de transactions et le montant moyen de chaque client sur 1, 7 et 30 jours.
5. La Card Precision@100 ne peut pas atteindre 1. Après le premier mois, il y a en moyenne 78 cartes frauduleuses par jour (entre 55 et 101) sur environ 3 800 cartes actives, et beaucoup ont déjà une fraude connue. Il faut donc lire les résultats par rapport au plafond d'un modèle parfait, calculé dans les mêmes conditions.

Le taux de fraude est le même la nuit et le jour, en semaine et le week-end (entre 0,84 % et 0,89 %), donc je n'ai pas mis de feature d'heure ou de jour, contrairement au livre.

</details>

<details>
<summary><b>Features et tests anti-fuite</b></summary>

`make features` calcule 12 features par transaction, en une vingtaine de secondes.

| Famille | Fenêtres | Contenu |
|---|---|---|
| Client | 1, 7 et 30 jours | nombre de transactions et montant moyen, transaction scorée comprise |
| Terminal | 1, 7 et 30 jours, décalées de 7 jours | nombre de transactions et taux de fraude |

Les features du terminal sont décalées à cause du délai d'étiquetage : une fraude n'est confirmée qu'après enquête, environ 7 jours plus tard. Une feature qui utiliserait les étiquettes des 7 derniers jours marcherait très bien hors ligne mais serait impossible à calculer en production. Deux tests le vérifient : les features d'une transaction ne changent pas quand on supprime toutes les transactions suivantes, ni quand on inverse les étiquettes pas encore connues à sa date.

</details>

<details>
<summary><b>PySpark</b></summary>

`src/fraudops/spark_features.py` calcule les mêmes features avec des fonctions de fenêtre Spark (`rangeBetween` sur le temps en secondes). Un test vérifie que les deux versions donnent les mêmes valeurs, et `make benchmark` compare les temps sur tout le jeu en revérifiant l'égalité sur les 1,8 M de transactions.

| Étape | Temps |
|---|---|
| pandas | 15 s |
| Spark, démarrage de la session | 9 s |
| Spark, calcul et écriture | 32 s |

Mesures faites sur une machine à 1 cœur et 4 Go de mémoire. À ce volume pandas va plus vite, parce que Spark paie le démarrage de la JVM et l'organisation de ses tâches sans rien pouvoir paralléliser sur un seul cœur. Spark devient utile quand les données ne tiennent plus en mémoire sur une machine, ou quand plusieurs cœurs ou machines se partagent le travail.

</details>

## Reproduire

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
make explain      # entraîne le modèle de production et calcule les valeurs SHAP
make drift        # PSI hebdomadaire des variables et du score
make api          # lance l'API sur http://127.0.0.1:8000
make docker       # construit et lance l'image Docker de l'API
make site         # génère le rapport statique dans _site/
make baf          # BAF, si data/baf/Base.csv a été téléchargé depuis Kaggle
```

## Limites

- Les données carte sont simulées : les fraudes suivent des règles connues, les scores ne se transposent pas tels quels à de vraies données, et le simulateur n'a aucune dérive.
- Les coûts sont des hypothèses (10 € par contrôle, le montant pour une fraude manquée). Les autres coûts, comme les litiges, la réémission de carte ou la confiance du client, ne sont pas comptés.
- Il n'y a pas de boucle de rétroaction : en production, une transaction bloquée ne révèle jamais sa vraie étiquette.

## Documentation

- [Rapport en ligne](https://mohamedradhi52.github.io/fraudops/), régénéré à chaque nouvelle publication de résultats
- [Cadrage métier](docs/cadrage.md) : coûts, capacité d'investigation, métriques
- [Journal des décisions](docs/DECISIONS.md) : les choix du projet et leur raison
- [Résultats BAF](reports/baf/RESULTS.md), publiés par le workflow

Sources : Le Borgne, Siblini, Lebichot et Bontempi, *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook*, Université libre de Bruxelles, 2022 (code sous GPL-3.0, pas repris ici). Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022 (données sous licence non commerciale, pas republiées).
