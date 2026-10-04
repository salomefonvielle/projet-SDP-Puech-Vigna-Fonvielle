# Projet SDP — Jalon 1

Côme-Alexis Puech, Hugo Vigna, Salomé Fonvielle — CentraleSupélec, mention IA, 2026-2027.

## Contenu

```
rapport_Jalon1.pdf            description du modèle (questions iv à viii bis) et résultats
src/mrsort.py                 implémentation (programme linéaire mixte, Gurobi)
data/dataset0.csv             jeu de données fourni (60 objets, 5 critères, 3 catégories)
resultats/modele_dataset0.txt paramètres du modèle appris sur Dataset0
```

## Modèle appris sur Dataset0

Restitution du learning set : 100 % (marge alpha = 0.071).

|            | crit1 | crit2 | crit3 | crit4 | crit5 |
|------------|-------|-------|-------|-------|-------|
| poids w    | 1/7   | 2/7   | 1/7   | 1/7   | 2/7   |
| frontière b1 (cat. 1/2) | 34.5 | 35 | 33 | 34.5 | 33 |
| frontière b2 (cat. 2/3) | 41.5 | 49 | 33 | 43.5 | 49 |

Seuil de majorité lambda = 0.643.

## Exécution

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # une licence Gurobi est nécessaire au-delà de 2000 variables

# apprendre un modèle sur Dataset0 (n et p sont déduits du fichier)
.venv/bin/python src/mrsort.py learn data/dataset0.csv

# protocole de test : modèle de référence aléatoire, apprentissage, accord sur 1000 objets test, 20 essais
.venv/bin/python src/mrsort.py test -n 4 -p 2 -m 50 --runs 20
```
