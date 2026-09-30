# Journal des décisions

Les choix structurants du projet, avec leur raison.

## Cadrage

- **Card Precision@100 comme métrique principale du jeu carte.** Elle mesure ce que l'équipe traite réellement chaque jour. L'accuracy et la ROC-AUC sont trompeuses avec moins de 1 % de fraude.
- **Coûts hypothétiques explicites.** Aucun coût réel n'est disponible : les hypothèses sont écrites dans `cadrage.md` et les résultats de coût sont toujours présentés avec elles.
- **Bootstrap sur les jours.** Les transactions d'une même journée sont corrélées ; les jours sont l'unité de rééchantillonnage.

## Données transactionnelles

- **Implémentation indépendante du simulateur du Handbook.** Le code du livre est sous GPL-3.0, incompatible avec une licence MIT. Le simulateur est réécrit à partir de la description du livre, avec les mêmes paramètres et les mêmes scénarios, et vectorisé avec NumPy.
- **Scénario 3 tiré transaction par transaction.** Le livre sélectionne exactement un tiers des transactions d'un client compromis ; ici chacune est frauduleuse avec une probabilité d'un tiers. L'effet est le même en moyenne et le code plus simple.
- **Colonnes du livre, en minuscules**, pour pouvoir comparer avec ses résultats.

## Features

- **Délai d'étiquetage de 7 jours pour les features terminal.** Une fraude n'est connue qu'après enquête : utiliser les étiquettes récentes serait une fuite qui gonfle les résultats hors ligne.
- **Pas de feature d'heure ni de jour.** L'exploration montre le même taux de fraude la nuit, le jour et le week-end dans ce simulateur.
- **Fenêtres définies à la seconde près.** Une fenêtre qui se termine à l'instant t contient toutes les transactions du client à cet instant, y compris celles de la même seconde, comme une fenêtre temporelle Spark.

## PySpark

- **pandas reste la version de référence.** Sur 1,8 M de transactions, pandas est plus rapide que Spark en local. La version Spark montre le passage à l'échelle, et un test garantit que les deux donnent les mêmes valeurs.
- **Fenêtres Spark sur le temps en secondes.** `rangeBetween` s'applique à une colonne numérique : avec des secondes entières, les bornes des fenêtres sont exactes.

## Environnement

- **Python 3.14, pandas 2.3 et numpy 2.4.** PySpark 4.2 ne supporte pas encore pandas 3, et numpy 2.5 provoque des avertissements de dépréciation dans pandas 2.3.
- **Java 21**, pris en charge par Spark 4 et disponible sur Ubuntu comme dans la CI.

## Validation et baselines

- **Ré-entraînement chaque semaine sur 28 jours.** La fenêtre contient environ 2 400 fraudes et suit l'évolution des terminaux compromis.
- **Cartes connues bloquées pendant les semaines de test**, comme dans le Handbook. Le modèle est jugé sur les fraudes que la banque ne connaît pas encore, ce qui divise par deux le nombre de fraudes à trouver.
- **Semaines de validation distinctes des semaines de test.** Règles, hyperparamètres, seuil et calibration sont choisis sur les 4 semaines de validation ; les 8 semaines de test ne servent qu'aux résultats.
- **Bootstrap de l'AUC-PR par pondération des jours.** Les transactions sont triées une seule fois, puis chaque tirage pondère les jours par leur nombre de tirages. Le résultat est identique à scikit-learn, une vingtaine de fois plus vite.

## LightGBM

- **Réglage restreint à 6 combinaisons**, départagées par l'AUC-PR de validation, plus stable que la Card Precision@100 sur 4 semaines. Chaque combinaison est suivie dans MLflow.
- **Ni pondération des classes ni SMOTE.** Le classement n'en a pas besoin, et les probabilités restent exploitables pour la calibration.
- **Entraînement déterministe** : les mêmes résultats quel que soit le nombre de cœurs de la machine.
- **Part de fraudes stoppées par scénario.** Elle montre que l'écart au plafond vient des fraudes qu'aucune donnée ne permet encore de repérer, pas d'un défaut du modèle.

## Seuil et coût

- **Coût calculé carte par carte et jour par jour.** Une carte contrôlée coûte 10 € ; une fraude coûte son montant si sa carte n'est ni contrôlée ce jour-là, ni déjà bloquée. Le blocage d'une carte évite ses fraudes suivantes, comme en production.
- **Seuil choisi sur la validation, jamais sur le test.** Les courbes de test servent seulement à vérifier que le seuil choisi tombe près du minimum.
- **Intervalle du coût par bootstrap des coûts journaliers**, comme pour les autres métriques.

## Calibration

- **Calibration isotonique ajustée sur les semaines de validation.** Elle ne suppose aucune forme de courbe, et la validation compte assez de fraudes (environ 1 200) pour l'estimer.
- **Règle de perte attendue plutôt qu'un seuil sur la probabilité.** Contrôler une carte dès que probabilité × montant dépasse 10 € : le seuil découle des coûts au lieu d'être cherché, et le montant en jeu compte.
- **Contre-exemple avec pondération des classes.** Le même LightGBM entraîné avec des poids équilibrés montre, chiffres à l'appui, pourquoi calibrer avant de décider.

## BAF

- **Données BAF jamais dans le dépôt.** Leur licence interdit l'usage commercial : GitHub Actions les télécharge avec le jeton Kaggle gardé en secret, publie les métriques et les figures par un commit automatique, et garde le modèle 90 jours comme artefact du workflow.
- **Seuil fixé sur le mois de validation, puis gardé sur le test.** C'est plus strict que le papier, qui fixe le seuil sur le test lui-même ; les deux mesures sont publiées.
- **Bootstrap sur les demandes**, indépendantes entre elles, et apparié entre modèles.
- **Colonnes catégorielles typées** plutôt qu'encodées en entiers : LightGBM les traite nativement et la régression logistique les encode en one-hot.

## Équité

- **Un seuil par groupe plutôt que le ThresholdOptimizer.** Fairlearn maximise un objectif sans pouvoir fixer le FPR global à 5 %, point de fonctionnement du protocole BAF. Les seuils sont donc calculés directement, un par groupe, à 5 % de FPR sur la validation.
- **Fairlearn pour l'audit, numpy pour le bootstrap.** MetricFrame prend environ une seconde par calcul sur 250 000 demandes ; les 1 000 tirages du ratio de FPR sont faits avec numpy, et un test vérifie que les deux donnent le même ratio.
- **Groupes du papier : 50 ans et plus contre les autres**, puis le détail par tranche d'âge.
- **Modèle sans l'âge entraîné à côté**, pour mesurer l'effet des variables qui portent la même information.

## Explicabilité

- **TreeSHAP intégré à LightGBM plutôt que la bibliothèque shap.** C'est le même algorithme, avec des valeurs exactes, et l'API n'a pas besoin d'une dépendance de plus.
- **Importance mesurée sur deux populations** : toutes les transactions, et celles que le modèle signale. La moyenne sur toutes les transactions sous-estime les variables décisives sur peu de cas, comme le risque du terminal.
- **Modèle de production versionné** (`models/lightgbm.txt`, 200 Ko) : l'API et l'image Docker l'utilisent tel quel.

## Dérive

- **Variables discrètes comparées valeur par valeur**, les continues sur les déciles de la référence : un découpage en déciles masquerait la dérive d'une variable binaire.
- **Score surveillé sur un modèle figé.** Chaque modèle ré-entraîné a sa propre échelle de scores ; comparer les scores de modèles différents donnerait des PSI jusqu'à 0,7 sans aucune dérive des données.
- **Seuils d'alerte usuels, 0,1 et 0,25**, chacun associé à une action.

## API

- **Features reçues déjà calculées.** En production, un feature store tient les fenêtres glissantes à jour ; l'API se contente de scorer, ce qui la garde simple et rapide.
- **Image minimale** : FastAPI, LightGBM et numpy, le modèle texte versionné, un utilisateur sans privilèges. Le coût d'un contrôle est recopié dans l'API, et un test vérifie qu'il reste égal à celui de l'étude.
- **Décision par perte attendue**, la règle retenue à l'étape de calibration : contrôler quand probabilité × montant dépasse 10 €.
