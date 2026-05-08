"""Outils de monitoring du modele en production : baseline + drift Evidently.

Le module est volontairement separe de ``src/`` parce que les scripts
sont des outils ops a lancer en CLI (jamais importes par le serving) :
``baseline_snapshot.py`` calcule la reference depuis le test set,
``run_drift_report.py`` genere les rapports Evidently HTML qui seront
servis dans la page Streamlit ``11_drift.py``.
"""
