# Cadrage métier

Les hypothèses avec lesquelles j'évalue les modèles. Les coûts ne sont pas des chiffres réels, je les ai regroupés ici pour pouvoir les discuter et les changer facilement.

## Deux problèmes, deux décisions

| | Fraude carte (Handbook) | Ouverture de compte (BAF) |
|---|---|---|
| Ce qui est scoré | chaque transaction | chaque demande d'ouverture |
| Décision | l'équipe contrôle chaque jour les cartes les plus suspectes | la demande est refusée ou envoyée en revue manuelle |
| Contrainte | 100 cartes contrôlées par jour | au plus 5 % de faux positifs |
| Métrique principale | Card Precision@100 | rappel à 5 % de FPR |

## Pourquoi pas l'accuracy

Moins de 1 % des transactions sont frauduleuses. Un modèle qui ne signale jamais rien a donc plus de 99 % d'accuracy et ne détecte aucune fraude. La ROC-AUC est trompeuse elle aussi : dominée par l'immense majorité de transactions légitimes, elle reste élevée même quand la plupart des alertes sont fausses. Les métriques retenues regardent plutôt combien de vraies fraudes il y a parmi les cas que l'équipe peut traiter.

## Fraude carte

Une équipe d'analystes contrôle au plus 100 cartes par jour. Chaque jour, les cartes sont classées par leur score le plus élevé de la journée et l'équipe traite les 100 premières. Le modèle ne bloque rien seul : il ordonne une file de travail. Une carte dont la fraude est confirmée est bloquée et sort de la file les jours suivants.

Une fraude n'est connue qu'après enquête ou réclamation du client. Hypothèse : l'étiquette d'une transaction est disponible 7 jours après. Les features et l'entraînement n'utilisent que les étiquettes connues à la date du score.

Hypothèses de coût :

| Événement | Coût retenu | Justification |
|---|---|---|
| Carte contrôlée | 10 € | environ 10 minutes d'analyste, appel au client compris |
| Fraude non détectée | montant de la transaction | la banque rembourse le client (paiement non autorisé, article L133-18 du Code monétaire et financier) |
| Fraude détectée | coût du contrôle | la carte est bloquée, les fraudes suivantes sont évitées |

Coût total sur une période : 10 € par carte contrôlée, plus la somme des montants frauduleux non détectés. Le seuil d'alerte est choisi pour minimiser ce coût sans dépasser 100 cartes par jour.

Non pris en compte : frais de litige entre banques, réémission de la carte, perte de confiance d'un client bloqué à tort ou victime d'une fraude.

## Ouverture de compte (BAF)

Le protocole du papier BAF fixe le seuil de façon à signaler 5 % des demandes légitimes (FPR de 5 %) et compare les modèles sur le rappel obtenu à ce seuil. L'hypothèse métier derrière cette contrainte : la banque accepte de refuser ou de retarder au plus 5 % de ses bons clients. Le seuil est fixé sur la période de validation puis appliqué tel quel sur la période de test.

L'âge est un attribut sensible. On mesure le ratio des FPR entre les plus de 50 ans et les autres, comme dans le papier : un ratio éloigné de 1 signifie que les bons clients d'un groupe sont plus souvent refusés que ceux de l'autre.

## Métriques retenues

| Métrique | Jeu | Rôle |
|---|---|---|
| Card Precision@100 | carte | part de vraies fraudes parmi les 100 cartes contrôlées chaque jour, en moyenne sur les jours |
| AUC-PR | carte, BAF | qualité du classement sans choisir de seuil, adaptée aux classes déséquilibrées |
| Coût total | carte | décision finale sous contrainte de capacité |
| Rappel à 5 % de FPR | BAF | métrique officielle du papier |
| Score de Brier, courbe de fiabilité | carte, BAF | qualité des probabilités (calibration) |
| Ratio de FPR par âge | BAF | équité |

Chaque résultat est donné avec un intervalle de confiance à 95 % obtenu par bootstrap : sur les jours pour le jeu carte, apparié entre modèles pour BAF. Les transactions d'une même journée, d'une même carte ou d'un même terminal ne sont pas indépendantes : rééchantillonner des jours donne des intervalles plus honnêtes que rééchantillonner des transactions.
