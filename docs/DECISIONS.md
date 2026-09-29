# Journal des décisions

Les choix structurants du projet, avec leur raison.

## Cadrage

- **Card Precision@100 comme métrique principale du jeu carte.** Elle mesure ce que l'équipe traite réellement chaque jour. L'accuracy et la ROC-AUC sont trompeuses avec moins de 1 % de fraude.
- **Coûts hypothétiques explicites.** Aucun coût réel n'est disponible : les hypothèses sont écrites dans `cadrage.md` et les résultats de coût sont toujours présentés avec elles.
- **Bootstrap sur les jours.** Les transactions d'une même journée sont corrélées ; les jours sont l'unité de rééchantillonnage.
