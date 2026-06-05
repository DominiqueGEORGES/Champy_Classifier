"""Page Streamlit présentant les limites assumées et les perspectives d'industrialisation.

Recense les axes identifiés mais non implémentés dans le temps imparti, et la
manière dont ils s'industrialiseraient. Pour chaque axe, la solution s'appuie sur
le socle déjà en place :
- scalabilité de l'API d'inférence ;
- portabilité et fork complet du projet et des données ;
- boucle qualité et réentraînement continu ;
- veille et intégration de nouveaux modèles ;
- orchestration du réentraînement via un DAG Airflow ;
- évaluation automatisée du modèle en production ;
- extension du référentiel d'espèces.

La page est purement rédactionnelle : elle ne fait aucun appel externe et
matérialise une trajectoire maîtrisée plutôt qu'une liste de manques.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Limites et perspectives",
    page_icon="🍄",
    layout="wide",
)

st.title("Limites et perspectives d'industrialisation")

st.markdown(
    """
Tout projet MLOps livre un socle, pas un produit figé. Cette page recense les axes
que nous avons identifiés sans pouvoir les implémenter dans le temps imparti, et
surtout la manière dont ils s'industrialiseraient. L'objectif n'est pas de lister
des manques, mais d'expliciter une trajectoire maîtrisée : pour chaque axe, la
solution est connue et s'appuie sur le socle déjà en place.
"""
)

st.divider()

st.subheader("Scalabilité (montée en charge)")
st.markdown(
    """
L'API d'inférence est sans état et le modèle est servi depuis un registre, ce qui
la rend horizontalement scalable par construction. La mise à l'échelle consisterait
à placer plusieurs instances de l'API derrière un répartiteur de charge, avec un
autoscaling piloté par la latence et le débit de requêtes, sous orchestrateur de
conteneurs. Le suivi Prometheus déjà en place fournit les métriques de décision.
"""
)

st.subheader("Portabilité et fork complet du projet")
st.markdown(
    """
Les données sont versionnées via DVC, mais le jeu brut de réentraînement représente
un volume important, aujourd'hui hébergé sur un remote unique. Pour permettre à
quiconque de forker le projet **et** ses données, la solution consiste à publier ce
remote sur un stockage objet compatible S3 dédié, à séparer clairement les artefacts
légers nécessaires à la démonstration des données brutes nécessaires au
réentraînement, et à documenter le chemin de récupération complet. Le lien vers le
remote DVC et la procédure de pull figurent dans la documentation d'installation.
"""
)

st.subheader("Boucle qualité et déclenchement du réentraînement")
st.markdown(
    """
La détection de dérive par Evidently est en place. L'étape suivante serait de fermer
la boucle : collecter les prédictions de production, mesurer leur qualité dès que des
retours fiables sont disponibles, et déclencher automatiquement un réentraînement
au-delà d'un seuil de dégradation. On passerait ainsi d'un monitoring d'observation à
un entraînement continu.
"""
)

st.subheader("Veille et intégration de nouveaux modèles")
st.markdown(
    """
L'arrivée régulière de nouvelles architectures appelle un protocole d'évaluation
systématique. Tout nouveau modèle candidat serait mesuré sur un jeu de validation
figé, comparé au modèle en production, et déployé en doublure ou en test progressif
avant toute bascule. L'adoption ne se ferait que sur un gain mesuré et significatif,
jamais sur la nouveauté seule.
"""
)

st.subheader("Orchestration du réentraînement (DAG)")
st.markdown(
    """
Le socle Airflow déjà déployé servirait de fondation à un graphe d'orchestration
dédié : récupération des données, curation, découpage, entraînement, évaluation,
enregistrement au registre, puis déploiement conditionné à une amélioration. Le
pipeline de curation existant, déjà reproductible, en constituerait la première
brique.
"""
)

st.subheader("Évaluation automatisée du modèle en production")
st.markdown(
    """
Une évaluation périodique sur un jeu de référence, avec suivi des métriques dans
MLflow et alerte en cas de dégradation, rendrait visible toute baisse de performance
dans le temps. La pertinence est ici nuancée : sur un référentiel d'espèces stable,
le modèle ne se dégrade pas spontanément. En revanche, sur un projet évolutif ou à
périmètre ouvert, cette surveillance automatisée devient indispensable.
"""
)

st.subheader("Extension du référentiel d'espèces")
st.markdown(
    """
Le périmètre actuel couvre 30 espèces. L'élargir, par exemple à d'autres aires
biogéographiques comme le continent africain, suppose d'étendre le référentiel,
de collecter des données étiquetées supplémentaires et de réentraîner avec une
stratégie adaptée : gestion du déséquilibre entre classes et, à plus grande échelle,
recours possible à une structure taxonomique hiérarchique.
"""
)
