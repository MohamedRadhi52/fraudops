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
