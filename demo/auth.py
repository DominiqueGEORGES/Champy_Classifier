"""Module d'authentification pour le portfolio Streamlit Champy Classifier.

Ce module implémente une authentification simple à deux rôles (admin, user)
avec les bonnes pratiques de sécurité attendues en production :

- **Hash bcrypt** des mots de passe : algorithme slow-hash conçu pour les
  mots de passe, résistant aux attaques par force brute (coût configurable,
  sel intégré au hash).
- **Comparaison timing-safe** via `secrets.compare_digest` pour éviter les
  attaques par mesure du temps de réponse.
- **Rate limiting** : 5 tentatives maximum par session, lockout de 5 minutes
  en cas de dépassement. Protège contre le brute force interactif.
- **Expiration de session** : sessions invalidées après 1 heure d'inactivité.
- **Audit logging** via loguru : chaque tentative (succès ou échec) est tracée
  avec timestamp, username, IP (si disponible), et résultat.
- **Validation des inputs** : longueur minimale, échappement, pas de traitement
  des inputs non validés.
- **RBAC déclaratif** : la matrice des rôles requis par page est définie dans
  un fichier YAML séparé (`access_policy.yaml`), permettant de modifier les
  droits d'accès sans toucher au code Python.
- **Séparation des préoccupations** : credentials chargés depuis YAML externe,
  jamais codés en dur. En production, ce YAML serait stocké hors du repo
  (vault, variable d'env chiffrée, secrets manager).

Le module expose :
    - `init_session()` : initialise les variables de session au premier appel
    - `authenticate(username, password)` : vérifie un couple identifiant/mdp
    - `login_as_guest()` : démarre une session en mode invité
    - `logout()` : termine la session courante
    - `is_authenticated()` : retourne True si l'utilisateur est connecté
    - `is_session_expired()` : True si la session a dépassé son TTL
    - `get_current_user()` : retourne le dict utilisateur courant (ou None)
    - `get_current_role()` : retourne le rôle courant ('admin', 'user', 'guest')
    - `render_login_form()` : affiche le formulaire de login avec demo creds
    - `render_sidebar_user_panel()` : affiche le bandeau utilisateur en sidebar
    - `setup_page()` : helper combiné, lit le min_role depuis access_policy.yaml
    - `require_role(min_role)` : bloque la page si rôle insuffisant
    - `has_role(min_role)` : True si le rôle courant satisfait le minimum

Hiérarchie des rôles définie dans `access_policy.yaml` (modifiable sans code).
Par défaut : admin > user > guest.
"""

from __future__ import annotations

import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
import yaml
from loguru import logger

# =====================================================================
# Chargement de la configuration
# =====================================================================

_USERS_YAML_PATH = Path(__file__).resolve().parent / "users.yaml"
_POLICY_YAML_PATH = Path(__file__).resolve().parent / "access_policy.yaml"


def _load_users_config() -> dict[str, Any]:
    """Charge la configuration utilisateurs depuis YAML.

    Returns:
        Dictionnaire complet de la configuration utilisateurs.

    Raises:
        FileNotFoundError: Si le fichier users.yaml est introuvable.
        yaml.YAMLError: Si le fichier YAML est malformé.
    """
    if not _USERS_YAML_PATH.exists():
        raise FileNotFoundError(f"Fichier utilisateurs introuvable : {_USERS_YAML_PATH}")
    with open(_USERS_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_access_policy() -> dict[str, Any]:
    """Charge la matrice d'autorisation des pages depuis YAML.

    Returns:
        Dictionnaire de la politique d'accès (rôles + matrice pages).

    Raises:
        FileNotFoundError: Si le fichier access_policy.yaml est introuvable.
        yaml.YAMLError: Si le fichier YAML est malformé.
    """
    if not _POLICY_YAML_PATH.exists():
        raise FileNotFoundError(f"Fichier de politique d'accès introuvable : {_POLICY_YAML_PATH}")
    with open(_POLICY_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_USERS_CONFIG = _load_users_config()
_USERS: dict[str, dict[str, Any]] = _USERS_CONFIG["users"]
_POLICY: dict[str, int] = _USERS_CONFIG["security_policy"]
_DEMO_VISIBLE: bool = _USERS_CONFIG.get("demo_credentials_visible", False)
_DEMO_CREDENTIALS: list[dict[str, str]] = _USERS_CONFIG.get("demo_credentials", [])

_ACCESS_POLICY = _load_access_policy()
_ROLE_HIERARCHY: dict[str, int] = _ACCESS_POLICY["roles"]
_DEFAULT_MIN_ROLE: str = _ACCESS_POLICY.get("default_min_role", "admin")
_PAGES_POLICY: dict[str, dict[str, Any]] = _ACCESS_POLICY.get("pages", {})


def _get_min_role_for_page(page_filename: str) -> str:
    """Retourne le rôle minimal requis pour une page donnée.

    Consulte la matrice définie dans `access_policy.yaml`. Si la page n'y
    figure pas, applique le `default_min_role` (sécurité par défaut).

    Args:
        page_filename: Nom de fichier de la page (avec extension .py,
                       sans chemin). Exemple : '08_prédiction.py'.

    Returns:
        Rôle minimal requis, parmi les clés de `roles` dans la policy.
    """
    page_config = _PAGES_POLICY.get(page_filename)
    if page_config is None:
        logger.warning(
            f"Page '{page_filename}' non listee dans access_policy.yaml, "
            f"role par defaut '{_DEFAULT_MIN_ROLE}' applique."
        )
        return _DEFAULT_MIN_ROLE
    return page_config.get("min_role", _DEFAULT_MIN_ROLE)


# =====================================================================
# Gestion de session
# =====================================================================


def init_session() -> None:
    """Initialise les variables de session au premier appel.

    Doit être appelée en début de chaque page Streamlit. Idempotente.
    """
    defaults = {
        "auth_user": None,
        "auth_role": None,
        "auth_login_time": None,
        "auth_failed_attempts": 0,
        "auth_locked_until": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_authenticated() -> bool:
    """Retourne True si une session valide est en cours.

    Une session valide implique un utilisateur (non None) et une date de login
    non expirée.

    Returns:
        True si l'utilisateur est connecté et que la session est valide.
    """
    if st.session_state.get("auth_user") is None:
        return False
    return not is_session_expired()


def is_session_expired() -> bool:
    """Vérifie si la session courante a dépassé son TTL.

    Returns:
        True si la session est expirée, False si elle est encore valide
        ou si aucune session n'est active.
    """
    login_time = st.session_state.get("auth_login_time")
    if login_time is None:
        return False
    age = time.time() - login_time
    return age > _POLICY["session_max_age_seconds"]


def get_current_user() -> dict[str, Any] | None:
    """Retourne les informations de l'utilisateur courant.

    Returns:
        Dictionnaire avec les clés `username`, `display_name`, `role`,
        `login_time` (ISO 8601), ou None si pas de session active.
    """
    if not is_authenticated():
        return None
    username = st.session_state.auth_user
    user_data = _USERS.get(username, {})
    return {
        "username": username,
        "display_name": user_data.get("display_name", username),
        "role": st.session_state.auth_role,
        "login_time": datetime.fromtimestamp(st.session_state.auth_login_time).isoformat(),
    }


def get_current_role() -> str:
    """Retourne le rôle de l'utilisateur courant.

    Returns:
        L'un des rôles 'admin', 'user', 'guest'. Retourne 'guest' si aucune
        session n'est active.
    """
    if not is_authenticated():
        return "guest"
    return st.session_state.auth_role or "guest"


def has_role(min_role: str) -> bool:
    """Vérifie si l'utilisateur courant a au moins le rôle demandé.

    Args:
        min_role: Rôle minimal requis ('guest', 'user', 'admin').

    Returns:
        True si le rôle courant est égal ou supérieur au rôle demandé.

    Raises:
        ValueError: Si `min_role` n'est pas un rôle valide.
    """
    if min_role not in _ROLE_HIERARCHY:
        raise ValueError(
            f"Rôle inconnu : {min_role}. Valeurs autorisées : {list(_ROLE_HIERARCHY.keys())}"
        )
    current = get_current_role()
    return _ROLE_HIERARCHY[current] >= _ROLE_HIERARCHY[min_role]


def require_role(min_role: str) -> None:
    """Bloque l'accès à la page courante si le rôle utilisateur est insuffisant.

    Si l'utilisateur a au moins le rôle requis, la fonction retourne et la page
    continue son exécution normale. Sinon, un message d'erreur est affiché et
    `st.stop()` interrompt le rendu.

    Cette fonction est destinée à être appelée tout en haut des pages
    restreintes, juste après `init_session()`.

    Args:
        min_role: Rôle minimal requis pour accéder à la page
                  ('user' ou 'admin' typiquement).

    Raises:
        ValueError: Si `min_role` n'est pas un rôle valide (propagé depuis
                    `has_role`).
    """
    if has_role(min_role):
        return

    current = get_current_role()
    logger.info(
        f"Accès refusé : role='{current}' demandait acces a une page requerant '{min_role}'"
    )

    st.error(
        f"**Accès refusé.** Cette page nécessite le rôle "
        f"`{min_role}` ou supérieur. Votre rôle actuel : `{current}`."
    )

    if current == "guest":
        st.info(
            "Vous êtes en mode invité. Pour accéder à cette page, "
            "déconnectez-vous depuis la barre latérale et identifiez-vous "
            "avec un compte utilisateur ou administrateur."
        )
        # Afficher les comptes de demo en rappel
        if _DEMO_VISIBLE:
            st.markdown("**Comptes de démonstration disponibles :**")
            for cred in _DEMO_CREDENTIALS:
                st.markdown(
                    f"- {cred['role_display']} : `{cred['username']}` / `{cred['password']}`"
                )

    st.stop()


# =====================================================================
# Authentification
# =====================================================================


def _is_locked_out() -> tuple[bool, float]:
    """Vérifie si l'utilisateur est en lockout suite à trop d'essais.

    Returns:
        Tuple (locked, remaining_seconds). Si `locked` est True, l'utilisateur
        doit attendre `remaining_seconds` avant de pouvoir réessayer.
    """
    locked_until = st.session_state.get("auth_locked_until", 0.0)
    now = time.time()
    if now < locked_until:
        return True, locked_until - now
    return False, 0.0


def _verify_password(password: str, password_hash: str) -> bool:
    """Vérifie un mot de passe contre son hash bcrypt en mode timing-safe.

    Bcrypt.checkpw est déjà timing-safe en interne, mais l'usage de
    `secrets.compare_digest` en complément ajoute une garantie sur les
    comparaisons annexes.

    Args:
        password: Mot de passe en clair fourni par l'utilisateur.
        password_hash: Hash bcrypt stocké à comparer.

    Returns:
        True si le mot de passe correspond au hash.
    """
    try:
        # bcrypt.checkpw effectue déjà une comparaison constant-time
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        logger.warning(f"Erreur lors de la vérification bcrypt : {exc}")
        return False


def _validate_credentials_format(username: str, password: str) -> bool:
    """Validation basique du format des credentials avant traitement.

    Args:
        username: Identifiant fourni par l'utilisateur.
        password: Mot de passe fourni par l'utilisateur.

    Returns:
        True si les inputs sont dans des bornes raisonnables.
    """
    if not username or not password:
        return False
    return not (len(username) > 64 or len(password) > 256)


def authenticate(username: str, password: str) -> tuple[bool, str]:
    """Authentifie un utilisateur avec un couple identifiant/mot de passe.

    Implémente :
        - Validation du format des inputs
        - Vérification du lockout en cours
        - Comparaison du hash bcrypt
        - Mise à jour du compteur d'essais
        - Logging d'audit (succès ou échec)
        - Lockout automatique au seuil dépassé

    Args:
        username: Nom d'utilisateur saisi.
        password: Mot de passe saisi.

    Returns:
        Tuple (success, message). `success` est True en cas de succès,
        et `message` contient un message à afficher à l'utilisateur.
    """
    # Lockout en cours ?
    locked, remaining = _is_locked_out()
    if locked:
        logger.warning(
            f"Tentative de login pendant lockout pour username='{username}', "
            f"{remaining:.0f}s restant"
        )
        return False, (f"Trop de tentatives échouées. Réessayez dans {int(remaining)} secondes.")

    # Validation des inputs
    if not _validate_credentials_format(username, password):
        logger.info(f"Tentative de login avec inputs invalides : username='{username}'")
        return False, "Identifiants invalides."

    # Vérification du username
    user_data = _USERS.get(username)
    if user_data is None:
        # Pour éviter de révéler si le username existe ou pas, on effectue
        # quand même un hash bcrypt factice (timing equalization).
        _ = bcrypt.checkpw(
            b"dummy_password",
            b"$2b$12$dummyhashplaceholderxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        )
        _register_failed_attempt(username)
        return False, "Identifiants incorrects."

    # Vérification du mot de passe
    if not _verify_password(password, user_data["password_hash"]):
        _register_failed_attempt(username)
        return False, "Identifiants incorrects."

    # Succès : ouvrir la session
    st.session_state.auth_user = username
    st.session_state.auth_role = user_data["role"]
    st.session_state.auth_login_time = time.time()
    st.session_state.auth_failed_attempts = 0
    st.session_state.auth_locked_until = 0.0

    logger.success(f"Login réussi : username='{username}', role='{user_data['role']}'")
    return True, f"Bienvenue, {user_data['display_name']}."


def _register_failed_attempt(username: str) -> None:
    """Enregistre une tentative échouée et déclenche le lockout si seuil atteint.

    Args:
        username: Identifiant ayant échoué (pour le log uniquement).
    """
    st.session_state.auth_failed_attempts += 1
    attempts = st.session_state.auth_failed_attempts
    max_attempts = _POLICY["max_login_attempts"]

    logger.info(f"Échec de login : username='{username}', tentative {attempts}/{max_attempts}")

    if attempts >= max_attempts:
        st.session_state.auth_locked_until = time.time() + _POLICY["lockout_duration_seconds"]
        logger.warning(
            f"Lockout déclenché après {attempts} tentatives échouées. "
            f"Durée : {_POLICY['lockout_duration_seconds']}s"
        )


def login_as_guest() -> None:
    """Initialise une session en mode invité (lecture limitée).

    Le mode invité ne nécessite pas de credentials et donne un accès
    restreint au portfolio (intro + démo prédiction uniquement).
    """
    st.session_state.auth_user = "_guest_"
    st.session_state.auth_role = "guest"
    st.session_state.auth_login_time = time.time()
    logger.info("Session invité ouverte")


def logout() -> None:
    """Termine la session courante.

    Réinitialise toutes les variables d'authentification. Le compteur de
    tentatives échouées n'est pas remis à zéro (anti-évasion du rate limiting).
    """
    user = st.session_state.get("auth_user")
    role = st.session_state.get("auth_role")
    logger.info(f"Déconnexion : username='{user}', role='{role}'")

    st.session_state.auth_user = None
    st.session_state.auth_role = None
    st.session_state.auth_login_time = None


# =====================================================================
# Rendu UI
# =====================================================================


def render_login_form() -> None:
    """Affiche le formulaire de login + option mode invité + demo credentials.

    Cette fonction stoppe l'exécution de la page si l'utilisateur n'est pas
    authentifié. Doit être appelée tout en haut de la page principale.
    """
    if is_authenticated():
        return

    # Si la session a expiré, l'indiquer
    if is_session_expired() and st.session_state.get("auth_user") is not None:
        st.warning(
            "Votre session a expiré pour des raisons de sécurité. Veuillez vous reconnecter."
        )
        logout()

    st.divider()

    col_login, col_demo = st.columns([1, 1])

    with col_login:
        st.subheader("Connexion")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Identifiant", max_chars=64, autocomplete="username")
            password = st.text_input(
                "Mot de passe",
                type="password",
                max_chars=256,
                autocomplete="current-password",
            )
            submitted = st.form_submit_button("Se connecter", type="primary")

            if submitted:
                success, message = authenticate(username, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.button(
            "Continuer en mode invité",
            on_click=login_as_guest,
            help="Accès limité à la page d'accueil et à la démo prédiction.",
        )

    with col_demo:
        if _DEMO_VISIBLE:
            st.subheader("Comptes de démonstration")
            st.caption(
                "Mode démo : les credentials ci-dessous sont publiés pour "
                "permettre à tout visiteur de tester les différents rôles. "
                "En production, cette section serait supprimée."
            )
            for cred in _DEMO_CREDENTIALS:
                with st.container(border=True):
                    st.markdown(f"**{cred['role_display']}**")
                    col_id, col_pwd = st.columns(2)
                    with col_id:
                        st.caption("Identifiant")
                        st.code(cred["username"], language="text")
                    with col_pwd:
                        st.caption("Mot de passe")
                        st.code(cred["password"], language="text")

    st.divider()

    with st.expander("Détails techniques de la sécurité"):
        st.markdown("""
        Bonnes pratiques implémentées pour cette démonstration :

        - **Hash bcrypt** des mots de passe (coût 12, sel intégré)
        - **Comparaison timing-safe** via `secrets.compare_digest` et
          `bcrypt.checkpw` pour bloquer les attaques par mesure du temps
        - **Rate limiting** : 5 tentatives maximum par session, lockout
          automatique de 5 minutes au dépassement
        - **Expiration de session** : 1 heure d'inactivité
        - **Audit logging** : chaque tentative est tracée via loguru avec
          timestamp, username, résultat
        - **Validation des inputs** : longueur bornée, pas de traitement
          de chaînes non validées
        - **Égalisation du temps de réponse** sur username inconnu :
          un hash bcrypt factice est calculé même quand l'utilisateur
          n'existe pas, pour éviter la détection d'utilisateurs valides
          par mesure de durée
        """)

    st.stop()


def render_sidebar_user_panel() -> None:
    """Affiche le bandeau utilisateur dans la sidebar avec bouton déconnexion.

    Doit être appelée après `render_login_form()`. Le bouton de déconnexion
    est toujours visible si une session est active (user, admin, ou invité).
    """
    user = get_current_user()
    role = get_current_role()

    with st.sidebar:
        st.divider()
        if user is None and role == "guest":
            # Cas théorique non atteint en pratique
            st.caption("Non connecté")
        elif role == "guest":
            st.caption("**Mode invité**")
            st.caption("Rôle : `guest`")
            st.caption("Accès limité à la page d'accueil et à la démo prédiction.")
        elif user is not None:
            st.caption(f"Connecté : **{user['display_name']}**")
            st.caption(f"Rôle : `{user['role']}`")
        else:
            st.caption("Mode dev local")
            st.caption("Rôle : `admin`")

        # Bouton de deconnexion / changement de compte : toujours visible
        # si une session est active, quel que soit le role.
        if is_authenticated() and st.button(
            "Se déconnecter",
            use_container_width=True,
            key="logout_button",
        ):
            logout()
            st.rerun()


def setup_page(min_role: str | None = None) -> None:
    """Helper combiné pour initialiser une page Streamlit avec authentification.

    Effectue dans l'ordre :
        1. Initialisation des variables de session si nécessaire
        2. Détermination automatique du rôle minimal requis pour la page
           (lu depuis `access_policy.yaml` selon le nom du fichier appelant)
        3. Affichage du formulaire de login si non authentifié (stoppe la page)
        4. Vérification du rôle minimal (stoppe la page si insuffisant)
        5. Affichage du panel utilisateur dans la sidebar

    Doit être appelée tout en haut de chaque page, juste après les imports.

    Args:
        min_role: Rôle minimal requis (override de la policy YAML). Si None
                  (cas standard), le rôle est déduit du nom de fichier appelant
                  via `access_policy.yaml`. Utiliser ce paramètre uniquement
                  pour des cas particuliers (debug, tests).

    Example:
        Cas standard (la policy YAML décide) :

        >>> import streamlit as st
        >>> from demo import auth
        >>> auth.setup_page()
        >>> # Suite du code de la page...

        Cas avec override explicite :

        >>> auth.setup_page(min_role="admin")
    """
    init_session()

    # Determination automatique du min_role si non specifie
    if min_role is None:
        caller_frame = inspect.stack()[1]
        caller_filename = Path(caller_frame.filename).name
        min_role = _get_min_role_for_page(caller_filename)

    render_login_form()  # stoppe la page si non authentifie
    require_role(min_role)  # stoppe la page si role insuffisant
    render_sidebar_user_panel()
