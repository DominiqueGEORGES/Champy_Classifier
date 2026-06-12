"""Page Streamlit : deploiement progressif (canari) orchestre par Airflow.

Support de presentation projetable pendant la soutenance. Le schema du flux est
embarque dans le fichier (aucune dependance externe), l'explication est
volontairement vulgarisee. A deposer dans demo/pages/ (ex : 13_deploiement_canari.py).
"""

import streamlit as st

st.set_page_config(page_title="Deploiement canari", layout="wide")

# Schema du flux, embarque en dur pour que la page soit autonome.
# Le style sur la balise svg le rend responsive a la largeur de la colonne.
SCHEMA_SVG = """<svg viewBox="0 0 820 700" style="width:100%;max-width:880px;height:auto" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#475569"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="820" height="700" fill="#ffffff"/>
  <text x="378" y="20" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">Deploiement canari du modele Champy</text>
  <path d="M300,75 L300,125" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <path d="M400,50 C470,50 500,112 543,147" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <path d="M300,175 L300,224" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <path d="M300,286 L300,329" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <path d="M300,391 L300,426" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <path d="M256,492 L169,568" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <path d="M344,492 L477,568" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <g font-size="11" fill="#475569">
    <rect x="306" y="92" width="74" height="17" rx="3" fill="#ffffff"/>
    <text x="343" y="104" text-anchor="middle">F1 &gt;= seuil</text>
    <rect x="444" y="90" width="66" height="17" rx="3" fill="#ffffff"/>
    <text x="477" y="102" text-anchor="middle">F1 &lt; seuil</text>
    <rect x="181" y="527" width="54" height="17" rx="3" fill="#ffffff"/>
    <text x="208" y="539" text-anchor="middle">adopter</text>
    <rect x="389" y="527" width="50" height="17" rx="3" fill="#ffffff"/>
    <text x="414" y="539" text-anchor="middle">rejeter</text>
  </g>
  <rect x="200" y="27" width="200" height="46" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
  <text x="300" y="55" text-anchor="middle" font-size="13" font-weight="600" fill="#1e3a8a">02 reentrainement</text>
  <rect x="547" y="121" width="176" height="58" rx="8" fill="#f3f4f6" stroke="#9ca3af" stroke-width="2"/>
  <text x="635" y="146" text-anchor="middle" font-size="13" font-weight="600" fill="#374151">arret</text>
  <text x="635" y="165" text-anchor="middle" font-size="11.5" fill="#6b7280">aucun deploiement</text>
  <rect x="180" y="127" width="240" height="46" rx="8" fill="#e0e7ff" stroke="#4f46e5" stroke-width="2"/>
  <text x="300" y="155" text-anchor="middle" font-size="13" font-weight="600" fill="#312e81">03 deploiement_progressif</text>
  <rect x="185" y="226" width="230" height="58" rx="8" fill="#eff6ff" stroke="#60a5fa" stroke-width="2"/>
  <text x="300" y="251" text-anchor="middle" font-size="13" font-weight="600" fill="#1e3a8a">appliquer_poids(10)</text>
  <text x="300" y="270" text-anchor="middle" font-size="11.5" fill="#3b82f6">api 90 / api_v2 10</text>
  <rect x="205" y="331" width="190" height="58" rx="8" fill="#eff6ff" stroke="#60a5fa" stroke-width="2"/>
  <text x="300" y="356" text-anchor="middle" font-size="13" font-weight="600" fill="#1e3a8a">alerte Discord</text>
  <text x="300" y="375" text-anchor="middle" font-size="11.5" fill="#3b82f6">canari actif</text>
  <polygon points="300,428 385,470 300,512 215,470" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="300" y="466" text-anchor="middle" font-size="13" font-weight="600" fill="#92400e">decision</text>
  <text x="300" y="484" text-anchor="middle" font-size="12" fill="#b45309">humaine</text>
  <rect x="45" y="571" width="240" height="58" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="165" y="596" text-anchor="middle" font-size="13" font-weight="600" fill="#14532d">05 full_new_model</text>
  <text x="165" y="615" text-anchor="middle" font-size="11.5" fill="#166534">api_v2 a 100, Production</text>
  <rect x="340" y="571" width="280" height="58" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="480" y="596" text-anchor="middle" font-size="13" font-weight="600" fill="#7f1d1d">06 restore_old_model</text>
  <text x="480" y="615" text-anchor="middle" font-size="11.5" fill="#991b1b">api a 100, candidat coupe</text>
</svg>"""

st.title("Déploiement progressif du modèle")
st.caption("Orchestration Airflow : réentraîner, déployer prudemment, laisser l'humain décider")

colonne_schema, colonne_texte = st.columns([3, 2], gap="large")

with colonne_schema:
    st.markdown(f'<div style="text-align:center">{SCHEMA_SVG}</div>', unsafe_allow_html=True)

with colonne_texte:
    st.markdown(
        """
**Airflow, le chef d'orchestre.** Il lance les bonnes tâches, dans le bon ordre,
au bon moment, et garde la trace de ce qui réussit ou échoue. Il ne fait pas le
travail lui-même, il fait entrer chaque étape au bon instant.

**Un DAG, une partition.** Chaque enchaînement de tâches est un DAG : une suite
d'étapes reliées par « ceci, puis cela », sans retour en arrière.

**Les quatre étapes :**

- **02, réentraînement.** On réapprend le modèle sur les données récentes et on
  mesure sa qualité. Trop faible, on s'arrête. Assez bon, on passe à la suite.
- **03, déploiement progressif.** On envoie seulement 10 % des utilisateurs vers
  le nouveau modèle, 90 % restent sur l'ancien, et on prévient un humain.
- **05, adoption.** On confie 100 % du trafic au nouveau modèle et on l'enregistre
  comme version officielle.
- **06, retour en arrière.** On débranche le nouveau modèle et on renvoie tout le
  monde sur l'ancien.
        """
    )

st.divider()

with st.expander("Pourquoi 10 % d'abord : le principe du canari"):
    st.markdown(
        """
Le nom vient des canaris qu'on descendait dans les mines : si l'air devenait
dangereux, l'oiseau réagissait avant les mineurs et donnait l'alerte à temps.
Même idée ici. On expose le nouveau modèle à une petite partie du trafic
seulement. S'il se comporte mal, seuls 10 % des utilisateurs sont touchés, et on
le voit avant de généraliser. Autre image : on fait goûter le plat à quelques
convives avant de servir toute la salle.
        """
    )

with st.expander("La décision reste humaine"):
    st.markdown(
        """
La machine ne promeut pas le modèle toute seule. Une fois le candidat à 10 %,
Airflow envoie une alerte sur Discord : « le candidat est en test, que
décide-t-on ? ». Un humain regarde, puis déclenche l'adoption ou le retour en
arrière. Sur quelques minutes de trafic, une décision automatique serait trop
hâtive pour un modèle tout neuf. Dans le métier, cela s'appelle garder un humain
dans la boucle.
        """
    )

with st.expander("Et en production, c'est branché ?"):
    st.markdown(
        """
L'orchestration et les quatre DAG sont écrits et fonctionnels. La bascule
effective du trafic au niveau de l'infrastructure, l'aiguillage qui répartit
réellement les utilisateurs entre les deux modèles, est conçue et documentée,
mais pas activée dans cette première version. Choix délibéré : ne pas toucher,
en pleine période d'examen, à la brique qui fait tourner toute la démonstration.
Les étapes pour l'activer figurent dans le document de conception.
        """
    )


# --- Detail des DAG : etapes clefs et code reel ---

SNIPPET_03 = r"""@task
def regler_trafic_canari() -> None:
    # nginx : basculer sur 90 / 10 (le candidat recoit 10 %)
    appliquer_poids(PART_CANARI)          # PART_CANARI = 10

@task
def alerter_humain() -> None:
    alerter_discord(
        "Canari actif : le candidat reçoit 10% du trafic. "
        "Décision attendue : promouvoir (05) ou revenir en arrière (06)."
    )

regler_trafic_canari() >> alerter_humain()
"""

SNIPPET_05 = r"""@task
def promouvoir() -> None:
    # nginx : basculer sur 100 / 0 (le candidat prend tout le trafic)
    appliquer_poids(100)

    # registre MLflow : le candidat passe en Production, l'ancien est archive
    client = MlflowClient()
    candidat = client.get_latest_versions("champy-classifier", stages=["Staging"])[0]
    client.transition_model_version_stage(
        "champy-classifier", candidat.version, "Production",
        archive_existing_versions=True,
    )

    alerter_discord(
        f"Modèle promu : version {candidat.version} en Production, 100% du trafic."
    )
"""

SNIPPET_06 = r"""@task
def revenir_en_arriere() -> None:
    # nginx : basculer sur 0 / 100 (candidat coupe, le champion reprend tout)
    appliquer_poids(0)
    alerter_discord(
        "Retour arrière : le candidat est coupé, le champion reçoit 100% du trafic."
    )
"""

SNIPPET_POIDS = r"""def appliquer_poids(part_candidat: int) -> None:
    # reecrit le bloc upstream nginx : champion / candidat
    bloc = (
        "upstream champy_api {\n"
        f"    server api:8000 weight={100 - part_candidat};\n"
        f"    server api_v2:8000 weight={part_candidat};\n"
        "}\n"
    )
    Path(NGINX_CONF).write_text(bloc, encoding="utf-8")
    # recharge nginx a chaud : il termine les requetes en cours avant de basculer
    docker.from_env().containers.get("champy_nginx").exec_run("nginx -s reload")
"""

SNIPPET_DISCORD = r"""def alerter_discord(message: str) -> None:
    webhook = os.environ["DISCORD_WEBHOOK_URL"]
    requests.post(webhook, json={"content": message}, timeout=10)
"""

st.divider()
st.subheader("Détail des DAG et code")
st.caption("Les étapes clés de chaque DAG, avec le code qui les exécute.")

onglet_03, onglet_05, onglet_06, onglet_module = st.tabs(
    ["03 canari", "05 adoption", "06 retour arrière", "Mécanique commune"]
)

with onglet_03:
    st.markdown(
        """
**Étapes clés**

1. nginx : basculer sur **90 / 10** (le candidat reçoit 10 % du trafic).
2. Alerter sur Discord et laisser la décision à l'humain.
        """
    )
    st.code(SNIPPET_03, language="python")

with onglet_05:
    st.markdown(
        """
**Étapes clés**

1. nginx : basculer sur **100 / 0** (le candidat prend tout le trafic).
2. Registre MLflow : passer le candidat en Production, archiver l'ancien.
3. Alerter sur Discord.
        """
    )
    st.code(SNIPPET_05, language="python")

with onglet_06:
    st.markdown(
        """
**Étapes clés**

1. nginx : basculer sur **0 / 100** (le candidat est coupé, le champion reprend tout).
2. Alerter sur Discord.
        """
    )
    st.code(SNIPPET_06, language="python")

with onglet_module:
    st.markdown(
        """
Les trois DAG s'appuient sur deux fonctions partagées.

**`appliquer_poids`** réécrit l'aiguillage nginx, qui dirige le trafic vers
l'ancien ou le nouveau modèle, puis recharge nginx sans coupure. Un poids de 0
retire un modèle du roulement.
        """
    )
    st.code(SNIPPET_POIDS, language="python")
    st.markdown("**`alerter_discord`** envoie le message sur le webhook du salon.")
    st.code(SNIPPET_DISCORD, language="python")
