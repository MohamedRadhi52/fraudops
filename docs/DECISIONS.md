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
