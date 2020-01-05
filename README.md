# TATIA

## BOND Adam
## VAILLANT-BEUCHOT Maël
=======
# Projet de TATIA
## Adam Bond - Maël Vaillant--Beuchot

## Résumé du projet
Dans le cadre du cours de TATIA en M1 Informatique nous avons mis en oeuvre un projet de d'analyse linguistique avec l'utilisation de concepts et d'outils vus durant le semestre.

Notre projet porte sur un outil d'analyse permettant de déterminer la probabilité que deux textes aient pu être écrits par le même auteur. Nous nous appuyons sur une analyse de différentes caractéristiques des textes (longueur des phrases, nombre de verbes par phrase, champs lexicaux etc.). Chacun de ces critères d'analyse ressort sous un facteur compris entre 0 et 1. Ces facteurs sont alors pondérés par des coefficients de pertinence basés sur nos tests et mis en commun pour obtenir un pourcentage final dans une première phase, et ils sont donnés à un algorithme de machine learning dans une second phase qui les analyse puis prédit si deux textes donnés sont du même auteur.

## Outils
Le projet est programmé en Python. Il utilise des outils d'analyse linguistique comme NLTK et SpaCy.
