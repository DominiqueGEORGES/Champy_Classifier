PROJET MLOPS LIORA "Champy Classifier, industrialisation MLOps d’un modèle de classification de champignons"


Etudiants :
	- FOCRAUD Loïc
	- GEORGES Dominique
	- PREGASSAME Saravana
	- SCHNEIDER Lionel


Contexte :
	Mise en production et cycle de vie MLOps d’un CNN Keras (ResNet50 fine-tuning) pour la reconnaissance de 30 espèces de champignons de France métropolitaine.


Résumé :

	Champy Classifier est une application Streamlit basée sur un modèle Deep Learning (CNN) en transfer learning et fine-tuning de ResNet50 (Keras).

	Le modèle classe des images de champignons parmi 30 espèces courantes en France métropolitaine, avec une accuracy moyenne de 95,6% et une précision moyenne de 96,2%.

	Aujourd’hui, l’application fonctionne en local mais n’est pas encore déployée en raison de contraintes de volumétrie (poids des artefacts, dataset d’images, limites GitHub sans LFS)
	et d’un manque d’industrialisation du cycle de vie : reproductibilité des entraînements, versioning des données et des modèles, automatisation des tests, packaging et déploiement,
	monitoring et ré-entraînement.


	Ce projet MLOps vise à transformer ce prototype en produit déployable et maintenable : structuration d’un pipeline de données et d’entraînement reproductible,
	mise en place d’un registre d’artefacts (modèles, métriques), CI/CD pour valider et livrer automatiquement, déploiement via conteners, et supervision (latence, taux d’erreur, …) avec des alertes.

	L’impact attendu est une réduction du temps de mise en production, une fiabilité accrue des releases, et un cadre d’exploitation permettant des mises à jour fréquentes et sûres du modèle.


Modèle existant :

	Modèle => CNN Keras basé sur ResNet50 (transfer learning + fine-tuning)

	Usage => classification d’images (photos) en 30 classes (espèces)

	Interface => application Streamlit (inférence + affichage résultats)

	Pré-traitements => pipeline de transformations/normalisation d’images (TensorFlow)

	Contexte de déploiement => prototype local, non déployé en production à ce stade


Objectif de l'approche MLOps :

	Le modèle est destiné à évoluer (nouvelles données, nouvelles espèces, amélioration des performances). Sans outillage MLOps, chaque mise à jour est risquée (régression, fuite de données, …),
	et le déploiement reste bloqué par la gestion des volumes. L’objectif est d’installer un cadre standard de production ML, réutilisable pour d’autres projets de vision.


Architecture du projet :

PROJET_MLOPS/
	├── .venv/
	└── Champy_Classifier/
		├── .dvc/	
     	│	├── .gitignore
		│	└── config
		├── data/
     	│	├── processed/
     	│	├── raw/
		│	├── split/
     	│	│	├── test/
     	│	│	├── train/
		│	│	├── val/
     	│	│	├── split_manifest.csv
		│	│	└── split_summary.csv
		│	├── champignons_france_top30.csv
		│	├── dataset_30_species.csv
		│	└── observations_mushrooms.csv
		├── models/
		├── notebooks/
		├── src/
		├── tests/
		├── .dvcignore
		├── .gitignore
		├── data.dvc
		├── models.dvc
		├── README.md
		└── requirements.txt
		

A FINIR DE COMPLETER

