"""Tests de non-régression de la vitrine publique (ADR 0001).

Ils verrouillent les règles de sécurité posées avant l'ouverture de la démo
au public : aucun secret versionné, aucun identifiant admin publié, aucune
page sans contrôle d'accès, et une façade publique limitée à la démo.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
NGINX = (ROOT / "configs/nginx/nginx.conf").read_text(encoding="utf-8")
USERS = yaml.safe_load((ROOT / "demo/users.yaml").read_text(encoding="utf-8"))
POLICY = yaml.safe_load((ROOT / "demo/access_policy.yaml").read_text(encoding="utf-8"))
PAGES = sorted((ROOT / "demo/pages").glob("*.py"))


# ---------------------------------------------------------------- secrets


def test_aucun_webhook_discord_en_dur() -> None:
    assert "discord.com/api/webhooks" not in COMPOSE
    assert "discordapp.com/api/webhooks" not in COMPOSE


@pytest.mark.parametrize(
    "motif",
    [
        r"PASSWORD:-admin\}",
        r"PASSWORD:-airflow\}",
        r"PASSWORD:-changeme",
        r"PASSWORD:-minioadmin\}",
    ],
)
def test_pas_de_mot_de_passe_par_defaut_faible(motif: str) -> None:
    assert re.search(motif, COMPOSE) is None, f"défaut faible trouvé : {motif}"


def test_la_demo_ne_recoit_aucun_identifiant_de_service() -> None:
    bloc_demo = COMPOSE.split("\n  demo:\n", 1)[1].split("\n  prometheus:\n", 1)[0]
    for variable in ("AIRFLOW_ADMIN_PASSWORD=", "GRAFANA_PASSWORD=", "MINIO_ROOT_PASSWORD="):
        assert variable not in bloc_demo


def test_seul_nginx_est_publie_sur_le_reseau() -> None:
    ports = re.findall(r'^\s+- "([\d.:]+)"', COMPOSE, flags=re.M)
    ouverts = [p for p in ports if not p.startswith("127.0.0.1:")]
    assert sorted(ouverts) == ["8088:80", "8089:8089"]


# ---------------------------------------------------------------- comptes


def test_compte_admin_desactive_dans_le_depot() -> None:
    assert USERS["users"]["admin"]["password_hash"] in ("", None)


def test_identifiants_admin_jamais_affiches() -> None:
    affiches = {c["username"] for c in USERS.get("demo_credentials", [])}
    assert "admin" not in affiches
    for compte in USERS.get("demo_credentials", []):
        role = USERS["users"][compte["username"]]["role"]
        assert role != "admin"


# ---------------------------------------------------------------- pages


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_chaque_page_appelle_setup_page(page: Path) -> None:
    assert "auth.setup_page(" in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_chaque_page_est_declaree_dans_la_politique(page: Path) -> None:
    assert page.name in POLICY["pages"]


def test_page_plateforme_reservee_et_sans_mot_de_passe() -> None:
    assert POLICY["pages"]["00_plateforme.py"]["min_role"] != "guest"
    source = (ROOT / "demo/pages/00_plateforme.py").read_text(encoding="utf-8")
    assert "_PASSWORD" not in source


# ---------------------------------------------------------------- nginx


def _serveur_public() -> str:
    debut = NGINX.index("listen 8089;")
    return NGINX[debut:]


def test_vitrine_ne_sert_que_la_demo() -> None:
    cibles = set(re.findall(r"proxy_pass\s+([^;]+);", _serveur_public()))
    assert cibles == {"http://demo:8501"}


def test_vitrine_limite_debit_et_taille() -> None:
    public = _serveur_public()
    assert "limit_req zone=public" in public
    assert "client_max_body_size" in public


def test_nginx_utilise_la_vraie_adresse_du_visiteur() -> None:
    assert "real_ip_header CF-Connecting-IP;" in NGINX
    assert "server_tokens off;" in NGINX


# ---------------------------------------------------------------- streamlit


def test_streamlit_durci() -> None:
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["client"]["showErrorDetails"] == "none"
    assert config["server"]["maxUploadSize"] <= 5
    assert config["server"]["runOnSave"] is False


# ---------------------------------------------------------------- auth


def test_surcharge_du_hash_par_variable_d_environnement() -> None:
    auth = pytest.importorskip("demo.auth")
    comptes = {"admin": {"role": "admin", "password_hash": ""}, "user": {"password_hash": "h"}}
    resultat = auth.apply_password_hash_overrides(
        comptes, {"CHAMPY_ADMIN_PASSWORD_HASH": "  $2b$12$abc  "}
    )
    assert resultat["admin"]["password_hash"] == "$2b$12$abc"
    assert resultat["user"]["password_hash"] == "h"
    assert comptes["admin"]["password_hash"] == "", "l'entrée ne doit pas être modifiée"


def test_compte_sans_hash_desactive() -> None:
    auth = pytest.importorskip("demo.auth")
    assert auth.is_account_enabled({"password_hash": "$2b$12$abc"}) is True
    assert auth.is_account_enabled({"password_hash": ""}) is False
    assert auth.is_account_enabled({"password_hash": None}) is False
    assert auth.is_account_enabled({}) is False
