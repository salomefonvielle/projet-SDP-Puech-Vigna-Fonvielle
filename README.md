# Projet SDP — Apprentissage de modèles MR-Sort

Côme-Alexis Puech, Hugo Vigna, Salomé Fonvielle — CentraleSupélec, mention IA, 2026-2027.

```
sujet/     énoncé du projet
rapport/   rapport LaTeX et PDF déposés sur Edunao (Jalon1_Puech_Vigna_Fonvielle.pdf)
src/       mrsort.py : apprentissage MR-Sort par programmation linéaire mixte (Gurobi)
data/      learning sets (exemple : 4 critères notés sur 20, 3 catégories)
tp0/       prise en main de Gurobi : exercices P1 à P4 du notebook TP0
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Une licence Gurobi (académique gratuite) est nécessaire au-delà de 2000 variables.

## Utilisation

Apprendre un modèle à partir d'un learning set CSV (n colonnes de performances, puis la classe ;
en-tête facultatif ; n et p sont déduits du fichier) :

```bash
.venv/bin/python src/mrsort.py learn data/exemple_learning_set.csv
```

Protocole de test du Jalon 1 (modèle de référence aléatoire, apprentissage, accord sur 1000 objets test,
répété 20 fois) :

```bash
.venv/bin/python src/mrsort.py test -n 4 -p 2 -m 50 --runs 20
```

## Rapport

Compilation : `cd rapport && latexmk -pdf -outdir=build rapport.tex` (le PDF est dans `rapport/build/`).
