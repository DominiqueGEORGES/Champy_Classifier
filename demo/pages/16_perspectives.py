"""Page Streamlit : ce qu'exige la mise en production d'un projet de prédiction en entreprise.

Cette page prend de la hauteur par rapport au cas Champy. Elle décrit ce qu'un
système de prédiction doit fournir et garantir pour passer en production dans un
environnement d'entreprise, et utilise Champy comme illustration concrète. Cinq
gros sujets structurent la démonstration : exploitation à l'échelle, automatisation
du cycle de vie du modèle, sécurité et gouvernance des accès, interface et
expérience, ouverture et extension. Pour chacun : le principe général, la manière
dont Champy le préfigure, les points d'attention, et un ou plusieurs schémas à la
charte du projet.

Chaque sujet est enfermé dans un bloc repliable (st.expander) afin que la page
s'ouvre sur la liste des sujets, comme une table des matières, et que chaque bloc
reste autonome et facile à retoucher. Chaque case des schémas nomme le composant
réel de la stack (NGINX, BentoML, MLflow, Prometheus, Evidently, Airflow, Streamlit,
MinIO, DagsHub) pour faire le lien avec l'architecture. La page est purement
rédactionnelle : aucun appel externe. Les schémas sont décrits en Graphviz et rendus
par Streamlit, sans dépendance ni accès réseau.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Mise en production et perspectives",
    page_icon="🍄",
    layout="wide",
)

st.title("Industrialiser un projet de prédiction en entreprise")

st.markdown(
    """
Un projet de prédiction qui tourne en démonstration n'est pas encore un produit
d'entreprise. Le passage en production impose un ensemble d'exigences : tenir la
charge, rester fiable dans le temps, sécuriser les accès, offrir une interface
exploitable, et rester reproductible. Cette page décrit ces exigences de manière
générale, puis montre comment Champy les préfigure et quels points d'attention
subsistent. L'objectif n'est pas de lister des manques, mais de démontrer la
maîtrise d'une trajectoire complète de mise en production.

**Ce qui est en place.** Le projet livre une chaîne complète, de la donnée à la
prédiction en ligne : un jeu de données curé et versionné (DVC), un modèle entraîné et
suivi (MLflow), exporté puis servi par une API sans état (BentoML), surveillé en continu
(Prometheus, Grafana, Evidently pour la dérive), alerté (Alertmanager, Discord), testé et
déployé automatiquement (CI/CD), exposé publiquement derrière NGINX et Cloudflare. Un
portail Streamlit de seize pages rend chaque étape visible.

**Ce qui reste à construire.** Pour un produit d'entreprise, cinq chantiers restent
ouverts, détaillés ci-dessous : tenir une forte charge avec mise à l'échelle automatique,
automatiser entièrement le réentraînement et le déploiement progressif, adosser les accès
à l'identité d'entreprise (SSO, rôles, clés pour les applications tierces), enrichir
l'interface d'un vrai mode administrateur, et rendre le projet pleinement portable et
extensible.

Le projet couvre déjà une grande variété d'aspects techniques. Les chantiers ci-dessous
n'ont pas été implémentés faute de temps et d'une infrastructure adéquate (pas de domaine
Active Directory, pas de cluster d'orchestration) ; leurs principes et leurs bases de
configuration sont définis et constituent notre feuille de route.

Chaque sujet est replié ci-dessous : la liste qui suit fait office de sommaire,
dépliez celui qui vous intéresse.
"""
)

st.divider()

# ============================================================================
# 1. Exploiter le service à grande échelle
# ============================================================================
with st.expander("1. Exploiter le service à grande échelle", expanded=False):
    st.markdown(
        """
Un service de prédiction en production doit absorber une charge variable et garder
une qualité stable dans le temps. Cela suppose une interface d'inférence **sans
état**, c'est-à-dire qui ne conserve rien en mémoire d'une requête à l'autre : on
peut alors en lancer autant de copies que nécessaire derrière un **répartiteur de
charge** (un aiguilleur qui distribue les requêtes), avec un ajustement automatique
du nombre de copies selon le trafic. En parallèle, la qualité des prédictions est
surveillée en continu pour repérer toute baisse.

Concrètement, ce pilotage revient à un **orchestrateur de conteneurs** comme
Kubernetes : il lance et arrête les copies selon le trafic (la mise à l'échelle
automatique), en garde quelques-unes préchauffées, et ne leur envoie des requêtes
qu'une fois prêtes grâce à des **sondes de disponibilité**. Cette brique n'est pas
implémentée dans Champy ; elle figure en pointillés sur le schéma comme la suite
logique.

Dans Champy, l'interface d'inférence (BentoML) est déjà sans état et le modèle est
servi depuis un **registre de modèles** (un catalogue versionné des modèles entraînés).
Les mesures de latence, de débit et de confiance sont collectées par Prometheus, l'outil
qui récupère et stocke ces mesures : c'est exactement la matière nécessaire pour piloter
une mise à l'échelle automatique.
"""
    )
    st.graphviz_chart(
        r"""
digraph echelle {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.45; ranksep=0.6; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  trafic [label="Trafic entrant", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  nginx  [label="Répartiteur de charge\n(NGINX)", fillcolor="#1F4E3D", fontcolor="#FAFAF5", color="#14352A", penwidth=1.6];
  subgraph cluster_api {
    label="Copies de l'API d'inférence (BentoML), ajustées selon le trafic"; labelloc="t";
    fontname="Helvetica"; fontsize=10; fontcolor="#57534E";
    style="rounded,filled"; fillcolor="#F6F4EE"; color="#C9C3B5";
    api1 [label="API\n(BentoML)"]; api2 [label="API\n(BentoML)"]; api3 [label="API\n(BentoML)"];
  }
  prom [label="Prometheus\nsurveille latence, débit, qualité", fillcolor="#F1EFE8", color="#A8A29E"];
  k8s [label="Orchestrateur de conteneurs\n(Kubernetes — piste non implémentée)\nlance, préchauffe et arrête les copies", style="rounded,dashed,filled", fillcolor="#F3F2EE", color="#9CA3AF", fontcolor="#6B7280"];
  trafic -> nginx;
  nginx -> api1; nginx -> api2; nginx -> api3;
  api1 -> prom [color="#A8A29E", style=dashed, arrowhead=none];
  api2 -> prom [color="#A8A29E", style=dashed, arrowhead=none];
  api3 -> prom [color="#A8A29E", style=dashed, arrowhead=none];
  k8s -> api2 [label="pilote le nombre de copies", style=dashed, color="#9CA3AF", fontcolor="#6B7280"];
}
""",
        use_container_width=True,
    )
    st.markdown(
        """
**Points d'attention**

- Le point de blocage se déplace souvent vers le chargement du modèle au démarrage de
  chaque copie : prévoir un démarrage préchauffé pour éviter les premières requêtes lentes.
- L'ajustement automatique doit se piloter sur des mesures métier (latence, débit) et
  non sur le seul taux d'occupation du processeur.
- Surveiller la qualité ne sert à rien si rien ne se déclenche : la surveillance doit
  être reliée à une alerte, puis au réentraînement (sujet 2).
"""
    )

# ============================================================================
# 2. Automatiser le cycle de vie du modèle
# ============================================================================
with st.expander("2. Automatiser le cycle de vie du modèle", expanded=False):
    st.markdown(
        """
Un modèle se dégrade quand la réalité observée s'éloigne des données sur lesquelles il
a appris : c'est la **dérive**. Un système mûr la détecte, sait relancer un entraînement,
l'orchestre de bout en bout, et n'adopte un nouveau modèle que sur preuve d'un gain.
On passe ainsi d'une simple observation à une amélioration continue. Ce sujet se lit en
deux temps : produire un modèle candidat, puis le mettre en production sans risque.
"""
    )

    st.markdown(
        """
#### Déclencher le réentraînement et produire un modèle candidat

Le modèle candidat peut avoir deux origines. La première est le **cycle de
réentraînement** : la dérive détectée par Evidently, l'outil qui compare la distribution
des prédictions récentes à une référence, signale qu'il faut réagir. La seconde est une
**voie d'approvisionnement** parallèle : un nouveau modèle entraîné hors cycle, par
exemple une architecture différente issue de la veille, se présente lui aussi comme
candidat.

Avant de lancer le cycle, un point de décision tranche : faut-il une **approbation
humaine** ? Si oui, le système attend un feu vert avant de lancer le DAG Airflow (dans
Champy, un simple message de validation reçu via Discord suffirait). Sinon, le
réentraînement **part seul**, en automatique. Airflow, le planificateur qui enchaîne les
tâches selon leurs dépendances, orchestre alors le cycle : récupération des données,
nettoyage et sélection (la **curation**), entraînement, évaluation. Quelle que soit
l'origine, on aboutit à un **modèle candidat versionné** au registre MLflow (le catalogue
versionné des modèles), prêt à être confronté à la production.
"""
    )
    st.graphviz_chart(
        r"""
digraph cycle_declenchement {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.45; ranksep=0.5; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  derive [label="Dérive détectée\n(Evidently)", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  nouveau [label="Entraînement d'un nouveau modèle\nhors cycle", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  decision [shape=diamond, label="Approbation\nhumaine\nrequise ?", style=filled, fillcolor="#F1EFE8", color="#A8A29E"];
  go [label="Réception du GO\n(Discord)", shape=note, fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  attente [label="Attendre la validation"];
  dag [label="DAG de réentraînement (Airflow)\ncuration, entraînement, évaluation"];
  candidat [label="Modèle candidat versionné\n(registre MLflow)", fillcolor="#1F4E3D", fontcolor="#FAFAF5", color="#14352A", penwidth=1.6];
  suite [label="→ déploiement progressif (ci-dessous)", shape=plaintext, fontcolor="#57534E"];
  derive -> decision;
  decision -> attente [label="oui"];
  decision -> dag [label="non\n(lancement automatique)"];
  go -> attente [style=dashed, color="#B45309", fontcolor="#7C2D12", label="débloque"];
  attente -> dag;
  dag -> candidat;
  nouveau -> candidat [label="voie d'approvisionnement", color="#B45309", fontcolor="#7C2D12"];
  candidat -> suite;
}
""",
        use_container_width=True,
    )
    st.markdown(
        """
**Points d'attention**

- Un réentraînement automatique sans garde-fou peut propager une donnée fautive :
  toujours valider sur un jeu figé avant de remplacer le modèle en production.
- L'approbation humaine est un choix de gouvernance : on l'active là où l'erreur coûte
  cher, on l'allège là où le cycle est éprouvé.
- Mesurer la vraie qualité demande des retours de terrain fiables, qui arrivent souvent
  en différé : ne pas confondre une dérive statistique avec une baisse réelle de performance.
"""
    )

    st.markdown(
        """
#### Déployer le candidat sans risque

Remplacer d'un coup le modèle en production, c'est du quitte ou double. La bonne
pratique est le **déploiement progressif**, souvent appelé « canary » : on fait d'abord
goûter le candidat à une petite part du trafic, par exemple 10 % des requêtes, le reste
continuant sur l'ancien. Pendant cette cohabitation, on **compare en continu** les deux
versions (qualité, latence, taux d'erreur). Si les indicateurs tiennent, on monte à
50 %, puis à 100 % : c'est la **promotion**. Si une **régression** apparaît sur la
nouvelle version, on fait marche arrière en quelques secondes : on **réactive la version
précédente** depuis le registre et on coupe la nouvelle. C'est tout l'intérêt d'un
registre versionné, l'ancienne version reste disponible à tout moment. Un répartiteur
comme NGINX suffit à réaliser ce partage : on y déclare les deux destinations avec un
poids. Dans Champy, le point d'entrée NGINX existe déjà ; il suffirait d'y faire tourner
une seconde instance BentoML servant le modèle candidat.
"""
    )
    st.graphviz_chart(
        r"""
digraph deploiement {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.5; ranksep=0.55; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  candidat [label="Modèle candidat\n(registre MLflow)", fillcolor="#1F4E3D", fontcolor="#FAFAF5", color="#14352A", penwidth=1.6];
  nginx [label="NGINX, répartiteur pondéré", fillcolor="#1F4E3D", fontcolor="#FAFAF5", color="#14352A", penwidth=1.6];
  candidat -> nginx [label="mise en service à faible part"];
  subgraph cluster_v1 {
    label="Ancien modèle (v1) — 90%"; labelloc="t"; fontname="Helvetica"; fontsize=10; fontcolor="#57534E";
    style="rounded,filled"; fillcolor="#F6F4EE"; color="#C9C3B5";
    api_v1 [label="API v1\n(BentoML)"];
  }
  subgraph cluster_v2 {
    label="Nouveau modèle (v2) — 10%"; labelloc="t"; fontname="Helvetica"; fontsize=10; fontcolor="#92400E";
    style="rounded,filled"; fillcolor="#FDF6EC"; color="#D8A65A";
    api_v2 [label="API v2\n(BentoML)", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12", penwidth=1.6];
  }
  nginx -> api_v1 [label="90%", penwidth=2.0];
  nginx -> api_v2 [label="10%", color="#B45309", fontcolor="#7C2D12"];
  prom [label="Prometheus compare v1 et v2\nqualité, latence, erreurs", fillcolor="#F1EFE8", color="#A8A29E"];
  api_v1 -> prom [style=dashed, color="#A8A29E", arrowhead=none];
  api_v2 -> prom [style=dashed, color="#A8A29E", arrowhead=none];
  regression [shape=diamond, label="v2\nrégresse ?", style=filled, fillcolor="#F1EFE8", color="#A8A29E"];
  prom -> regression [style=dashed, color="#A8A29E"];
  rollback [shape=note, label="Marche arrière rapide\nréactiver v1 depuis le registre,\ncouper v2", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  promo [label="Promotion\nv2 monte à 100%"];
  regression -> rollback [label="oui", color="#B45309", fontcolor="#7C2D12"];
  regression -> promo [label="non (indicateurs OK)"];
}
""",
        use_container_width=True,
    )
    st.markdown(
        "La configuration NGINX qui réalise ce partage du trafic tient dans un groupe de "
        "destinations pondérées. Sur dix requêtes, neuf vont à l'ancien modèle, une au nouveau :"
    )
    st.code(
        """# nginx.conf - les deux versions du modele servies en parallele
upstream champy_api {
    server api_v1:8000 weight=9;   # ancien modele : 9 requetes sur 10
    server api_v2:8000 weight=1;   # nouveau modele : 1 requete sur 10
}

server {
    listen 80;
    location /api/ {
        proxy_pass http://champy_api/;
        proxy_set_header Host $http_host;
    }
}""",
        language="nginx",
    )
    st.markdown(
        "Pour monter à 50/50, on met le même poids des deux côtés. Pour la marche arrière, "
        "NGINX n'accepte pas un poids nul : on coupe le nouveau avec la directive `down`."
    )
    st.code(
        """upstream champy_api {
    server api_v1:8000;
    server api_v2:8000 down;   # le nouveau ne recoit plus aucun trafic
}""",
        language="nginx",
    )
    st.markdown(
        "La bascule et la marche arrière s'appliquent à chaud, sans couper les requêtes en cours :"
    )
    st.code("docker compose exec nginx nginx -s reload", language="bash")
    st.markdown(
        """
**Points d'attention**

- Pour comparer les deux modèles, le suivi Prometheus doit étiqueter quel modèle a
  répondu à chaque requête.
- La détection de régression doit être automatique et reliée à la marche arrière :
  repérer trop tard une mauvaise version annule le bénéfice du déploiement progressif.
- Tester la marche arrière avant d'en avoir besoin : un rollback qui n'a jamais été
  répété n'est pas un filet de sécurité fiable.
- Faire tourner deux modèles en parallèle double la mémoire consommée pendant la
  transition : le prévoir.
"""
    )

# ============================================================================
# 3. Sécuriser et gouverner les accès
# ============================================================================
with st.expander("3. Sécuriser et gouverner les accès", expanded=False):
    st.markdown(
        """
En entreprise, un service de prédiction ne gère pas ses comptes dans son coin : il
s'adosse à un **module central de gestion des identités et des accès**, un SSO appuyé sur
l'annuaire d'entreprise (Active Directory). On se connecte une seule fois, et c'est
l'appartenance à un **groupe** de l'annuaire qui décide des droits, sans identifiant
propre à chaque outil. Ce module se place devant l'ensemble de la stack : le portail
Streamlit, l'API, mais aussi MLflow, Grafana, Airflow et MinIO.

Tous les outils ne savent pas déléguer leur authentification de la même façon. Trois le
font nativement (Grafana, Airflow, MinIO, ainsi que l'API) : ils interrogent directement
le SSO. Deux ne le savent pas vraiment (le portail Streamlit et MLflow) : on les place
derrière un **proxy d'authentification** qui vérifie l'identité avant de laisser passer.
L'accès public actuel de Champy, assuré par Cloudflare Access, est une première forme de
ce proxy. Ce module central n'est pas implémenté : il suppose un domaine Active Directory
que nous n'avons pas en environnement de projet ; ses principes et sa configuration sont
définis.
"""
    )
    st.graphviz_chart(
        r"""
digraph securite {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.45; ranksep=0.55; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  users [label="Utilisateurs (par rôle)\nlecteur, opérateur, administrateur", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  iam [label="Gestion des identités et des accès\nSSO / Active Directory — les groupes portent les droits\n(cible entreprise, non implémentée)", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12", penwidth=1.6];
  natif [label="Outils à authentification native\nGrafana, Airflow, MinIO, API (BentoML)"];
  proxy [label="Proxy d'authentification"];
  derriere [label="Outils protégés par le proxy\nPortail (Streamlit), MLflow"];
  secrets [label="Gestionnaire de secrets", shape=note, fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  audit [label="Journal d'audit (RGPD)", shape=note, fillcolor="#F1EFE8", color="#A8A29E", fontcolor="#57534E"];
  users -> iam;
  iam -> natif [label="connexion déléguée"];
  iam -> proxy; proxy -> derriere;
  secrets -> iam [style=dashed, color="#B45309", arrowhead=none, constraint=false];
  iam -> audit [style=dashed, color="#A8A29E", arrowhead=none, constraint=false];
}
""",
        use_container_width=True,
    )
    st.markdown(
        """
#### Ouvrir l'API à des applications tierces

Le service de prédiction n'a pas vocation à rester réservé à l'interface interne :
d'autres applications peuvent vouloir l'appeler directement, pour prédire ou pour relever
des statistiques. Cela ne s'ouvre pas en grand. On intercale une **couche de
permissions** : chaque application reçoit une **clé d'accès**, cette clé porte une
**portée de droits** précise (prédire, mais pas administrer), et un **quota** limite le
volume d'appels. On sait ainsi qui appelle quoi, on coupe une clé compromise sans toucher
aux autres, et on peut plafonner ou facturer l'usage.
"""
    )
    st.graphviz_chart(
        r"""
digraph acces_tiers {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.45; ranksep=0.55; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  app1 [label="Application tierce\n(prédiction)", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  app2 [label="Application tierce\n(statistiques)", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  gate [label="Contrôle des accès\nauthentifie l'appelant, vérifie la portée\nclés, quotas\n(prévu, non implémenté)", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12", penwidth=1.6];
  api [label="API de prédiction\n(BentoML)"];
  canal [label="Canal chiffré HTTPS / TLS\nclé ou certificat client (mTLS)\nni écoute ni altération en transit", shape=note, fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  app1 -> gate [label="HTTPS"]; app2 -> gate [label="HTTPS"];
  gate -> api;
  canal -> gate [style=dashed, color="#B45309", arrowhead=none, constraint=false];
}
""",
        use_container_width=True,
    )
    st.markdown(
        """
**Sécuriser les appels : authentifier l'appelant et chiffrer le canal**

Ouvrir l'API ne suffit pas, il faut aussi garantir deux choses : que l'appelant est bien
celui qu'il prétend être, et que personne ne peut écouter ni modifier les échanges en
chemin (l'attaque dite de l'**homme du milieu**). Deux briques répondent à cela. D'abord,
tout passe en **HTTPS (TLS)** : les échanges sont chiffrés de bout en bout, illisibles et
inaltérables en transit. Ensuite, chaque appel porte une preuve d'identité : au minimum
une **clé d'API** transmise dans l'en-tête, et pour une garantie plus forte qu'une clé
(qui peut être volée), un **certificat client (mTLS)** que l'appelant doit présenter.
Côté code, la vérification de l'appelant tient en quelques lignes :
"""
    )
    st.code(
        """# Verification de l'appelant a chaque appel : cle d'API dans l'en-tete
from fastapi import Header, HTTPException

CLES_AUTORISEES = {"app-predict", "app-stats"}  # une cle par application tierce

async def verifier_appelant(authorization: str = Header(...)):
    cle = authorization.removeprefix("Bearer ").strip()
    if cle not in CLES_AUTORISEES:
        raise HTTPException(status_code=401, detail="appelant non autorise")
""",
        language="python",
    )
    st.markdown(
        "Pour exiger en plus un certificat client (authentification mutuelle, mTLS), deux "
        "directives NGINX suffisent à refuser tout appelant sans certificat valide :"
    )
    st.code(
        """ssl_verify_client on;            # exige un certificat client valide (mTLS)
ssl_client_certificate ca.crt;   # autorite qui a signe les certificats des appelants
""",
        language="nginx",
    )
    st.markdown(
        "Ces mécanismes sont définis mais non implémentés, faute de temps : les principes "
        "et la configuration sont posés, l'activation reste à faire."
    )
    st.markdown(
        """
**Points d'attention**

- Savoir qui se connecte (**authentification**) ne suffit pas : il faut définir ce que
  chacun a le droit de faire (**autorisation**), par rôle et par clé.
- **Moindre privilège** : n'accorder que les droits strictement nécessaires, jamais plus.
- **Chiffrement** en transit (les échanges réseau) et au repos (les données stockées) ;
  faire tourner régulièrement (**rotation**) les secrets et les clés d'accès.
- Un secret ne doit jamais transiter par le dépôt de code ni par les journaux ; en
  entreprise, un gestionnaire de secrets dédié s'impose.
- **Isolation** : cloisonner les composants pour qu'une brèche sur l'un n'ouvre pas tout
  le reste.
- Conformité RGPD : tracer qui accède à quoi (journal d'audit), fixer une durée de
  conservation, prévoir l'effacement des données personnelles. Les droits attachés aux
  données elles-mêmes sont traités au sujet 5.
- Un accès en lecture seule pour la démonstration ne doit pas ouvrir les actions
  sensibles comme déclencher un réentraînement ou télécharger les données.
"""
    )

# ============================================================================
# 4. Soigner l'interface et l'expérience
# ============================================================================
with st.expander("4. Soigner l'interface et l'expérience", expanded=False):
    st.markdown(
        """
L'interface conditionne l'adoption, mais il faut d'abord être clair sur son rôle : le
portail Streamlit sert à **démontrer** et à **administrer**, jamais à rendre le service.
Les prédictions de production passent par l'**API (BentoML)**, appelée directement par
les applications consommatrices (voir le sujet 3) ; l'interface, elle, sert à montrer,
comprendre et piloter. Au-delà de la démonstration, un usage en entreprise réclame une
ergonomie soignée, des visualisations compréhensibles par des non-spécialistes, et un
**mode administrateur** pour les opérations courantes : déclencher un réentraînement,
comparer puis promouvoir un modèle, consulter l'état du système, sans passer par la ligne
de commande.

Dans Champy, le portail Streamlit expose déjà seize pages couvrant les données,
l'entraînement, la prédiction, la supervision et l'infrastructure, avec un principe
constant : aucune valeur n'est écrite en dur dans les pages, tout est lu en direct depuis
les sources. L'étape suivante serait un véritable mode administrateur en libre-service et
une interface adaptée à chaque rôle.
"""
    )
    st.graphviz_chart(
        r"""
digraph interface {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.4; ranksep=0.55; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  lecteur [label="Lecteur", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  operateur [label="Opérateur", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  admin [label="Administrateur", shape=ellipse, fillcolor="#F1EFE8", color="#A8A29E"];
  portail [label="Portail\n(Streamlit)\ndémonstration et administration", fillcolor="#1F4E3D", fontcolor="#FAFAF5", color="#14352A", penwidth=1.6];
  consulter [label="Consulter\ndonnées et prédictions"];
  operer [label="Superviser\nrelancer un service"];
  administrer [label="Administrer\ndéclencher un réentraînement,\npromouvoir un modèle", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  canal [shape=note, label="Le service de production passe par l'API (BentoML),\npas par le portail", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  lecteur -> portail; operateur -> portail; admin -> portail;
  portail -> consulter; portail -> operer; portail -> administrer;
  portail -> canal [style=dashed, color="#B45309", arrowhead=none, constraint=false];
}
""",
        use_container_width=True,
    )
    st.markdown(
        """
**Points d'attention**

- Une interface d'administration doit être protégée par les rôles (lien direct avec le
  sujet 3) : déclencher un réentraînement n'est pas une action de simple visiteur.
- Les actions longues, comme un réentraînement, doivent s'exécuter en arrière-plan avec
  un retour d'avancement, et non derrière un bouton qui fige l'écran.
- Lisibilité pour les non-spécialistes : un verdict en clair vaut mieux qu'un tableau de
  mesures brutes, comme déjà appliqué sur la page de dérive.
- Ne pas confondre interface et service : une application qui a besoin de prédictions
  appelle l'API, pas le portail. Si l'interface tombe, le service ne doit pas s'arrêter.
"""
    )

# ============================================================================
# 5. Ouvrir, porter et étendre
# ============================================================================
with st.expander("5. Ouvrir, porter et étendre", expanded=False):
    st.markdown(
        """
Un projet mûr est reproductible et transférable : on doit pouvoir le récupérer, le faire
tourner ailleurs sans dépendance cachée, et élargir son périmètre. Une nuance importante
d'emblée : le **code** et le **pipeline** se partagent librement, mais les **données** pas
toujours. Dans Champy, les images proviennent de sources externes (observatoires
naturalistes) aux licences variables ; rien ne garantit qu'on puisse les redistribuer
publiquement à tout le monde. Le fork « complet » porte donc d'abord sur le code et la
recette de fabrication ; reconstituer le jeu d'images peut imposer de repasser par les
sources d'origine.

Côté outillage, le code est versionné par Git et les données comme les modèles le sont par
DVC, l'outil qui gère les gros fichiers à côté de Git. Le projet fonctionne dans deux
modes : l'un appuyé sur un stockage distant partagé (DagsHub), l'autre entièrement local
et souverain (MinIO). Pour un fork, on publie le stockage des données sur un espace de
**stockage objet compatible S3** (le standard de stockage de fichiers du cloud), on sépare
les artefacts légers utiles à la démonstration des données lourdes nécessaires au
réentraînement, et on documente le chemin de récupération de bout en bout. Élargir le
périmètre, par exemple de trente espèces à d'autres aires géographiques, suppose de
collecter des données étiquetées supplémentaires, de gérer le déséquilibre entre classes,
et à grande échelle d'adopter une organisation hiérarchique : classer d'abord par grande
famille, puis par espèce.
"""
    )
    st.graphviz_chart(
        r"""
digraph ouverture {
  rankdir=TB; bgcolor="transparent"; pad=0.3; nodesep=0.45; ranksep=0.55; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#A8A29E", fillcolor="#F1EFE8", fontcolor="#1A1A1A", penwidth=1.2];
  edge [fontname="Helvetica", fontsize=10, color="#1F4E3D", fontcolor="#1A1A1A"];
  repo [label="Dépôt\nGit (code) + DVC (données et modèles)", fillcolor="#1F4E3D", fontcolor="#FAFAF5", color="#14352A", penwidth=1.6];
  partage [label="Mode partagé\nstockage distant (DagsHub)"];
  local [label="Mode local\n(MinIO souverain)"];
  fork [label="Reproductible ailleurs\nfork du code et du pipeline", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  droits [shape=note, label="Redistribution des images\nsoumise aux licences des sources", fillcolor="#FBEFD9", color="#B45309", fontcolor="#7C2D12"];
  repo -> partage; repo -> local;
  partage -> fork; local -> fork;
  fork -> droits [style=dashed, color="#B45309", arrowhead=none, constraint=false];
}
""",
        use_container_width=True,
    )
    st.markdown(
        """
**Points d'attention**

- Les droits sur les données priment sur l'envie de tout ouvrir : vérifier la licence de
  chaque source avant toute redistribution. Partager le code n'autorise pas à partager les
  images.
- Séparer dès le départ les données lourdes de réentraînement des artefacts légers de
  démonstration évite d'imposer un téléchargement de plusieurs dizaines de gigaoctets à
  qui veut seulement essayer.
- Une dépendance cachée (un chemin local, un secret non documenté) casse la
  reproductibilité : valider l'installation depuis un dossier vierge.
- Étendre le référentiel sans stratégie d'équilibrage dégrade les classes les moins
  représentées.
"""
    )
