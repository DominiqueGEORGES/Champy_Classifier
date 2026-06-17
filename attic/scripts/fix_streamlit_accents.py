"""Utilitaire ponctuel : restaure les accents français dans les pages Streamlit.

Ne modifie que les tokens STRING et COMMENT (et donc aussi les docstrings,
qui sont des STRING au sens de tokenize). Les identifiants Python (noms de
variables, fonctions, attributs) sont préservés pour éviter de casser le code.

Approche : parcours des tokens via tokenize, remplacement des substrings
aux spans (byte offsets) des tokens STRING/COMMENT, puis réécriture du
fichier. Le reste du code source est préservé à l'octet près.

Usage :
    python scripts/fix_streamlit_accents.py

Ce script est destiné à être exécuté une seule fois puis supprimé
(ou conservé comme trace pour la reproductibilité).
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

# Mapping des tokens non-accentués vers leur version française correcte.
# Utilise des limites de mots \b pour éviter les sous-chaînes parasites.
# Ordre : patterns plus longs avant les plus courts, capitalisés avant
# minuscules pour les règles où les deux coexistent.
REPLACEMENTS: list[tuple[str, str]] = [
    ("Donnees brutes", "Données brutes"),
    ("donnees brutes", "données brutes"),
    ("Donnees", "Données"),
    ("donnees", "données"),
    ("Donnee", "Donnée"),
    ("donnee", "donnée"),
    ("Entrainement", "Entraînement"),
    ("entrainement", "entraînement"),
    ("Entraine", "Entraîné"),
    ("entraine", "entraîné"),
    ("Evaluation", "Évaluation"),
    ("evaluation", "évaluation"),
    ("Prediction", "Prédiction"),
    ("prediction", "prédiction"),
    ("Predictions", "Prédictions"),
    ("predictions", "prédictions"),
    ("Predites", "Prédites"),
    ("predites", "prédites"),
    ("Predit", "Prédit"),
    ("predit", "prédit"),
    ("Especes", "Espèces"),
    ("especes", "espèces"),
    ("Espece", "Espèce"),
    ("espece", "espèce"),
    ("Metriques", "Métriques"),
    ("metriques", "métriques"),
    ("Metrique", "Métrique"),
    ("metrique", "métrique"),
    ("Metadonnees", "Métadonnées"),
    ("metadonnees", "métadonnées"),
    ("Apres", "Après"),
    ("apres", "après"),
    ("Verification", "Vérification"),
    ("verification", "vérification"),
    ("Verifier", "Vérifier"),
    ("verifier", "vérifier"),
    ("Verifiez", "Vérifiez"),
    ("verifiez", "vérifiez"),
    ("Reponse", "Réponse"),
    ("reponse", "réponse"),
    ("Reference", "Référence"),
    ("reference", "référence"),
    ("Genere", "Génère"),
    ("genere", "génère"),
    ("Generer", "Générer"),
    ("generer", "générer"),
    ("Generee", "Générée"),
    ("generee", "générée"),
    ("Generees", "Générées"),
    ("generees", "générées"),
    ("Generation", "Génération"),
    ("generation", "génération"),
    ("Modele", "Modèle"),
    ("modele", "modèle"),
    ("Modeles", "Modèles"),
    ("modeles", "modèles"),
    ("Integre", "Intègre"),
    ("integre", "intègre"),
    ("Repartition", "Répartition"),
    ("repartition", "répartition"),
    ("Repertoire", "Répertoire"),
    ("repertoire", "répertoire"),
    ("Repertoires", "Répertoires"),
    ("repertoires", "répertoires"),
    ("Deterministe", "Déterministe"),
    ("deterministe", "déterministe"),
    ("Deja", "Déjà"),
    ("deja", "déjà"),
    ("Degres", "Degrés"),
    ("degres", "degrés"),
    ("Desequilibre", "Déséquilibre"),
    ("desequilibre", "déséquilibre"),
    ("Strategie", "Stratégie"),
    ("strategie", "stratégie"),
    ("Etat", "État"),
    ("etat", "état"),
    ("Etapes", "Étapes"),
    ("etapes", "étapes"),
    ("Etape", "Étape"),
    ("etape", "étape"),
    ("Presence", "Présence"),
    ("presence", "présence"),
    ("Detail", "Détail"),
    ("detail", "détail"),
    ("Detection", "Détection"),
    ("detection", "détection"),
    ("Demonstration", "Démonstration"),
    ("demonstration", "démonstration"),
    ("Demarre", "Démarré"),
    ("demarre", "démarré"),
    ("Rafraichir", "Rafraîchir"),
    ("rafraichir", "rafraîchir"),
    ("Dependance", "Dépendance"),
    ("dependance", "dépendance"),
    ("Echoue", "Échoué"),
    ("echoue", "échoué"),
    ("Echouee", "Échouée"),
    ("echouee", "échouée"),
    ("Methode", "Méthode"),
    ("methode", "méthode"),
    ("Methodes", "Méthodes"),
    ("methodes", "méthodes"),
    ("Sante", "Santé"),
    ("sante", "santé"),
    ("Cles", "Clés"),
    ("cles", "clés"),
    ("Parametre", "Paramètre"),
    ("parametre", "paramètre"),
    ("Parametres", "Paramètres"),
    ("parametres", "paramètres"),
    ("Hyperparametres", "Hyperparamètres"),
    ("hyperparametres", "hyperparamètres"),
    ("Hyperparametre", "Hyperparamètre"),
    ("hyperparametre", "hyperparamètre"),
    ("Recupere", "Récupère"),
    ("recupere", "récupère"),
    ("Recuperer", "Récupérer"),
    ("recuperer", "récupérer"),
    ("Trouvee", "Trouvée"),
    ("trouvee", "trouvée"),
    ("Trouvees", "Trouvées"),
    ("trouvees", "trouvées"),
    ("Normalisee", "Normalisée"),
    ("normalisee", "normalisée"),
    ("Chargees", "Chargées"),
    ("chargees", "chargées"),
    ("Critere", "Critère"),
    ("critere", "critère"),
    ("Criteres", "Critères"),
    ("criteres", "critères"),
    ("Specifie", "Spécifié"),
    ("specifie", "spécifié"),
    ("Specifique", "Spécifique"),
    ("specifique", "spécifique"),
    ("Partages", "Partagés"),
    ("partages", "partagés"),
    ("Cachees", "Cachées"),
    ("cachees", "cachées"),
    ("Liees", "Liées"),
    ("liees", "liées"),
    ("Liee", "Liée"),
    ("liee", "liée"),
    ("Aleatoire", "Aléatoire"),
    ("aleatoire", "aléatoire"),
    ("Aleatoires", "Aléatoires"),
    ("aleatoires", "aléatoires"),
    ("Augmentees", "Augmentées"),
    ("augmentees", "augmentées"),
    ("Exporte", "Exporté"),
    ("exporte", "exporté"),
    ("Optimisee", "Optimisée"),
    ("optimisee", "optimisée"),
    ("Affichee", "Affichée"),
    ("affichee", "affichée"),
    ("Affichees", "Affichées"),
    ("affichees", "affichées"),
    ("Affiche", "Affiché"),
    ("Uploadee", "Uploadée"),
    ("uploadee", "uploadée"),
    ("Enregistree", "Enregistrée"),
    ("enregistree", "enregistrée"),
    ("Configuree", "Configurée"),
    ("configuree", "configurée"),
    ("Ecarte", "Écarte"),
    ("ecarte", "écarte"),
    ("Evolue", "Évolue"),
    ("evolue", "évolue"),
    ("Evolution", "Évolution"),
    ("evolution", "évolution"),
    ("Preconfigures", "Préconfigurés"),
    ("preconfigures", "préconfigurés"),
    ("Pre-configures", "Pré-configurés"),
    ("pre-configures", "pré-configurés"),
    ("Ameliore", "Améliore"),
    ("ameliore", "améliore"),
    ("Ameliorer", "Améliorer"),
    ("ameliorer", "améliorer"),
    ("Installe", "Installé"),
    ("installe", "installé"),
    ("Execution", "Exécution"),
    ("execution", "exécution"),
    ("Execute", "Exécute"),
    ("execute", "exécute"),
    ("Effectuee", "Effectuée"),
    ("effectuee", "effectuée"),
    ("Effectuees", "Effectuées"),
    ("effectuees", "effectuées"),
    ("Definie", "Définie"),
    ("definie", "définie"),
    ("Generalisation", "Généralisation"),
    ("generalisation", "généralisation"),
    ("Requete", "Requête"),
    ("requete", "requête"),
    ("Requetes", "Requêtes"),
    ("requetes", "requêtes"),
    ("Retournee", "Retournée"),
    ("retournee", "retournée"),
    ("Accumule", "Accumulé"),
    ("accumule", "accumulé"),
    ("Necessaires", "Nécessaires"),
    ("necessaires", "nécessaires"),
    ("Necessaire", "Nécessaire"),
    ("necessaire", "nécessaire"),
    ("Automatise", "Automatisé"),
    ("automatise", "automatisé"),
    ("Cote", "Côté"),
    ("cote", "côté"),
    ("Stratifie", "Stratifié"),
    ("stratifie", "stratifié"),
    ("Prouve", "Prouvé"),
    ("prouve", "prouvé"),
    ("Calcules", "Calculés"),
    ("calcules", "calculés"),
    ("Qualite", "Qualité"),
    ("qualite", "qualité"),
    ("Ecrite", "Écrite"),
    ("ecrite", "écrite"),
    ("Selectionnees", "Sélectionnées"),
    ("selectionnees", "sélectionnées"),
    ("Selectionne", "Sélectionne"),
    ("selectionne", "sélectionne"),
    ("Selection", "Sélection"),
    ("selection", "sélection"),
    ("Icone", "Icône"),
    ("icone", "icône"),
    ("Inference", "Inférence"),
    ("inference", "inférence"),
    ("Lateral", "Latéral"),
    ("lateral", "latéral"),
    ("Predefinis", "Prédéfinis"),
    ("predefinis", "prédéfinis"),
    ("Metadonnee", "Métadonnée"),
    ("metadonnee", "métadonnée"),
    ("Retournement", "Retournement"),  # no-op
    ("Redimensionne", "Redimensionne"),  # no-op
]

# Compile une regex par (old, new) avec limites de mots Unicode-friendly.
# On utilise (?<![A-Za-zÀ-ÿ0-9_]) et (?![A-Za-zÀ-ÿ0-9_]) pour éviter les
# correspondances à l'intérieur de mots contenant des accents adjacents.
_COMPILED = [
    (
        re.compile(r"(?<![A-Za-zÀ-ÿ0-9_])" + re.escape(old) + r"(?![A-Za-zÀ-ÿ0-9_])"),
        new,
    )
    for old, new in REPLACEMENTS
    if old != new
]


def apply_replacements(text: str) -> str:
    """Applique tous les remplacements à une chaîne.

    Args:
        text: Texte source (typiquement un token STRING ou COMMENT).

    Returns:
        Texte avec les accents restaurés.
    """
    for pattern, new in _COMPILED:
        text = pattern.sub(new, text)
    return text


def fix_source(source: str) -> str:
    """Réécrit la source : remplace uniquement dans les tokens STRING/COMMENT.

    Args:
        source: Code Python source complet.

    Returns:
        Code Python avec accents restaurés dans les chaînes et commentaires.
    """
    # tokenize.generate_tokens donne les (type, string, start, end, line).
    # On collecte les spans (start_offset, end_offset) pour les STRING/COMMENT
    # et on reconstruit la source en remplaçant uniquement ces spans.
    lines = source.splitlines(keepends=True)

    # Précalcule l'offset du début de chaque ligne dans la source.
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def pos_to_offset(row: int, col: int) -> int:
        """Convertit (row, col) tokenize (1-indexed row) en offset absolu."""
        return line_offsets[row - 1] + col

    spans: list[tuple[int, int]] = []
    readline = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            start_off = pos_to_offset(tok.start[0], tok.start[1])
            end_off = pos_to_offset(tok.end[0], tok.end[1])
            spans.append((start_off, end_off))

    # Reconstruit la source morceau par morceau (remplacement uniquement dans
    # les spans, le reste préservé à l'octet près).
    result: list[str] = []
    cursor = 0
    for start, end in spans:
        result.append(source[cursor:start])
        result.append(apply_replacements(source[start:end]))
        cursor = end
    result.append(source[cursor:])
    return "".join(result)


def main() -> None:
    """Point d'entrée : traite tous les fichiers demo/**/*.py."""
    root = Path(__file__).resolve().parent.parent
    demo_dir = root / "demo"
    files = sorted(demo_dir.rglob("*.py"))
    changed = 0
    for f in files:
        original = f.read_text(encoding="utf-8")
        try:
            fixed = fix_source(original)
        except tokenize.TokenizeError as e:
            print(f"[SKIP] {f} : {e}", file=sys.stderr)
            continue
        if fixed != original:
            # Écrit en UTF-8 sans BOM, newlines LF.
            f.write_text(fixed, encoding="utf-8", newline="\n")
            print(f"[FIX]  {f.relative_to(root)}")
            changed += 1
        else:
            print(f"[OK]   {f.relative_to(root)}")
    print(f"\n{changed} fichier(s) modifié(s) sur {len(files)}.")


if __name__ == "__main__":
    main()
