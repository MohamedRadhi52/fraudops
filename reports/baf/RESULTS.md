# Résultats sur BAF

Généré par le workflow `baf.yml` sur la variante Base, qui n'est jamais publiée.

## Performance

| Modèle | Rappel, seuil fixé en validation | FPR obtenu sur le test | Rappel à 5 % de FPR mesuré sur le test |
|---|---|---|---|
| Régression logistique | 49,5 % [47,7 ; 51,4] | 5,0 % [4,9 ; 5,1] | 49,6 % [47,7 ; 51,5] |
| LightGBM | 57,4 % [55,5 ; 59,2] | 5,9 % [5,8 ; 6,0] | 54,0 % [52,1 ; 55,8] |

## Équité entre groupes d'âge

| Variante | FPR, 50 ans et plus | FPR, moins de 50 ans | Ratio de FPR | Rappel |
|---|---|---|---|---|
| Seuil unique | 13,7 % | 4,4 % | 0,32 [0,31 ; 0,33] | 57,4 % [55,5 ; 59,2] |
| Un seuil par groupe d'âge | 6,4 % | 6,0 % | 0,94 [0,90 ; 0,98] | 55,5 % [53,6 ; 57,3] |
| Seuil unique, modèle sans l'âge | 11,1 % | 4,8 % | 0,44 [0,42 ; 0,45] | 56,7 % [54,9 ; 58,7] |

Rappel perdu avec un seuil par groupe : 1,9 % [0,7 ; 3,1].

## Dérive mois par mois, contre les mois d'entraînement

| Période | PSI du score | Variables à surveiller | Variables à ré-entraîner |
|---|---|---|---|
| mois 0 | 0,012 | 4 | 2 |
| mois 1 | 0,009 | 0 | 1 |
| mois 2 | 0,009 | 0 | 1 |
| mois 3 | 0,014 | 2 | 1 |
| mois 4 | 0,005 | 1 | 2 |
| mois 5 | 0,005 | 5 | 3 |
| mois 6 | 0,001 | 1 | 6 |
| mois 7 | 0,006 | 1 | 6 |
