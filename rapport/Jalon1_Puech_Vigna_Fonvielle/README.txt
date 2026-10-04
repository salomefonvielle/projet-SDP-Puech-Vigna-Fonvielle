Projet SDP - Jalon 1
Côme-Alexis Puech, Hugo Vigna, Salomé Fonvielle

Contenu :
- rapport_Jalon1.pdf : description du modèle et résultats
- src/mrsort.py : le code (Python + Gurobi)
- data/dataset0.csv : dataset0
- resultats/modele_dataset0.txt : paramètres du modèle appris sur dataset0

Installation : pip install -r requirements.txt (gurobipy, numpy)

Apprendre le modèle sur dataset0 :
    python src/mrsort.py learn data/dataset0.csv

Lancer le protocole de test (20 essais) :
    python src/mrsort.py test -n 4 -p 2 -m 50 --runs 20
