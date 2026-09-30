# Journal des décisions

Les principaux choix du projet, avec à chaque fois la raison.

## Cadrage

- J'ai pris la Card Precision@100 comme métrique principale pour la fraude carte, parce qu'elle mesure ce que l'équipe traite vraiment chaque jour. Avec moins de 1 % de fraude, l'accuracy et la ROC-AUC sont trompeuses.
- Je n'ai pas de vrais coûts, donc j'ai écrit mes hypothèses dans `cadrage.md` et je les rappelle à chaque résultat de coût.
- Le bootstrap se fait sur les jours, parce que les transactions d'une même journée sont corrélées.

## Données transactionnelles

- Le code du Handbook est sous GPL-3.0, pas compatible avec une licence MIT. J'ai donc réécrit le simulateur à partir de la description du livre, avec les mêmes paramètres et les mêmes scénarios, en vectorisant avec NumPy.
- Pour le scénario 3, le livre prend exactement un tiers des transactions d'un client compromis. Chez moi, chaque transaction est frauduleuse avec une probabilité d'un tiers, ce qui donne le même effet en moyenne avec un code plus simple.
- J'ai gardé les noms de colonnes du livre, en minuscules, pour pouvoir comparer avec ses résultats.

## Features

- Les features du terminal ont un délai de 7 jours : une fraude n'est connue qu'après enquête, et utiliser les étiquettes récentes serait une fuite.
- Pas de feature d'heure ni de jour, l'exploration montre le même taux de fraude la nuit, le jour et le week-end.
- Une fenêtre qui finit à l'instant t contient toutes les transactions du client à ce moment-là, y compris celles de la même seconde, comme une fenêtre temporelle Spark.

## PySpark

- pandas reste la version de référence, il va plus vite que Spark en local sur 1,8 M de lignes. La version Spark montre le passage à l'échelle, et un test vérifie que les deux donnent les mêmes valeurs.
- Les fenêtres Spark utilisent le temps en secondes, parce que `rangeBetween` veut une colonne numérique. Ça donne des bornes exactes.

## Environnement

- Python 3.14 avec pandas 2.3 et numpy 2.4. PySpark 4.2 ne supporte pas encore pandas 3, et numpy 2.5 provoque des avertissements de dépréciation avec pandas 2.3.
- Java 21, supporté par Spark 4 et disponible sur Ubuntu comme dans la CI.

## Validation et baselines

- Le modèle est ré-entraîné chaque semaine sur 28 jours, ce qui fait environ 2 400 fraudes par fenêtre et suit l'évolution des terminaux compromis.
- Pendant les semaines de test, les cartes déjà connues comme fraudées sont bloquées, comme dans le Handbook. Ça divise par deux le nombre de fraudes à trouver, mais le modèle est jugé sur ce que la banque ne sait pas encore.
- Règles, hyperparamètres, seuil et calibration sont choisis sur les 4 semaines de validation. Les 8 semaines de test ne servent qu'aux résultats.
- Pour aller plus vite, le bootstrap de l'AUC-PR trie les transactions une seule fois, puis pondère chaque jour par le nombre de fois où il est tiré. Le résultat est le même qu'avec scikit-learn, environ 20 fois plus vite.

## LightGBM

- Le réglage se limite à 6 combinaisons, départagées par l'AUC-PR de validation, plus stable que la Card Precision@100 sur 4 semaines. Tout est suivi dans MLflow.
- Pas de pondération des classes ni de SMOTE : le classement n'en a pas besoin, et les probabilités restent utilisables.
- L'entraînement est déterministe, pour avoir les mêmes résultats quel que soit le nombre de cœurs.
- J'ai mesuré la part de fraudes stoppées par scénario pour comprendre l'écart au plafond. Il vient des fraudes qu'aucune donnée ne permet encore de repérer.

## Seuil et coût

- Le coût est calculé carte par carte et jour par jour. Un contrôle coûte 10 €, et une fraude coûte son montant si la carte n'est ni contrôlée ce jour-là ni déjà bloquée. Bloquer une carte évite ses fraudes suivantes.
- Le seuil est choisi sur la validation. Les courbes du test servent juste à vérifier qu'il tombe près du minimum.
- L'intervalle du coût vient d'un bootstrap des coûts journaliers.

## Calibration

- La calibration isotonique est ajustée sur la validation. Elle ne suppose rien sur la forme de la courbe, et il y a assez de fraudes (environ 1 200) pour l'estimer.
- Plutôt qu'un seuil sur la probabilité, je contrôle une carte dès que probabilité × montant dépasse 10 €. Le seuil vient des coûts au lieu d'être cherché, et le montant compte.
- Le contre-exemple avec pondération des classes montre pourquoi il faut calibrer avant de décider.

## BAF

- Les données BAF ne sont jamais dans le dépôt (licence non commerciale). GitHub Actions les télécharge avec le jeton Kaggle gardé en secret, publie les métriques et les figures par un commit automatique, et garde le modèle 90 jours comme artefact.
- Le seuil est fixé sur le mois de validation puis gardé sur le test. C'est plus strict que le papier, qui fixe le seuil sur le test, donc je publie les deux mesures.
- Le bootstrap se fait sur les demandes, indépendantes entre elles, et il est apparié entre modèles.
- Les colonnes catégorielles sont typées plutôt qu'encodées en entiers. LightGBM les gère directement, et la régression logistique les encode en one-hot.

## Équité

- J'utilise un seuil par groupe plutôt que le `ThresholdOptimizer` de Fairlearn, qui ne permet pas d'imposer 5 % de faux positifs au global (le point de fonctionnement du protocole BAF).
- Fairlearn sert à l'audit et numpy au bootstrap. MetricFrame met environ une seconde par calcul sur 250 000 demandes, trop lent pour 1 000 tirages, et un test vérifie que les deux donnent le même ratio.
- Je garde les groupes du papier, 50 ans et plus contre les autres, avec en plus le détail par tranche d'âge.
- Un modèle sans l'âge est entraîné à côté, pour mesurer l'effet des variables qui portent la même information.

## Explicabilité

- J'utilise le TreeSHAP intégré à LightGBM plutôt que la librairie shap. C'est le même algorithme, les valeurs sont exactes, et l'API a une dépendance de moins.
- L'importance est mesurée sur toutes les transactions et sur celles que le modèle signale. La moyenne sur tout sous-estime les variables décisives sur peu de cas, comme le risque du terminal.
- Le modèle de production est versionné (`models/lightgbm.txt`, 200 Ko), et l'API comme l'image Docker l'utilisent tel quel.

## Dérive

- Les variables discrètes sont comparées valeur par valeur, les continues sur les déciles de la référence. Avec des déciles, on raterait la dérive d'une variable binaire.
- Le score est suivi sur un modèle figé. Chaque modèle ré-entraîné a sa propre échelle de scores, et les comparer donnerait des PSI jusqu'à 0,7 sans aucune dérive des données.
- J'utilise les seuils d'alerte classiques, 0,1 et 0,25, avec une action pour chacun.

## API

- L'API reçoit les features déjà calculées. En production, un feature store tiendrait les fenêtres à jour, et l'API ne fait que scorer.
- L'image est minimale : FastAPI, LightGBM, numpy, le modèle texte et un utilisateur sans privilèges. Le coût d'un contrôle est recopié dans l'API, et un test vérifie qu'il reste le même que dans l'étude.
- La décision suit la règle de perte attendue retenue à l'étape de calibration.

## Rapport

- La page est générée à partir des résultats publiés, avec juste la librairie standard. Aucun chiffre n'est recopié à la main, et un test vérifie que la page se remplit en entier.
- Le workflow Pages se relance après chaque exécution de BAF pour suivre les nouveaux résultats.
- Le README a deux parties : l'essentiel en haut, le détail dans des blocs repliables.
