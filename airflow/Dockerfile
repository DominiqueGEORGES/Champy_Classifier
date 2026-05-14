# Image Airflow custom pour Champy Classifier.
#
# Étend l'image officielle apache/airflow:2.10.3-python3.11 (LTS, support
# long terme) avec les bibliothèques nécessaires aux DAGs Champy :
#
#   - mlflow        : pour interroger DagsHub MLflow depuis les tasks
#   - python-dotenv : pour charger les variables d'environnement
#   - requests      : pour appels HTTP (santé API, notifications)
#
# Toute task qui aurait besoin de PyTorch ou de bibliothèques lourdes
# (entraînement, inférence) doit déléguer l'exécution à un autre poste
# via SSH ou Docker exec, plutôt que d'alourdir cette image.

FROM apache/airflow:2.10.3-python3.11

# Installation des dépendances système (curl pour healthcheck)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Retour à l'utilisateur airflow pour pip install (image officielle l'exige)
USER airflow

# Installation des libs Python supplémentaires
RUN pip install --no-cache-dir \
        "mlflow>=2.14,<3.0" \
        "python-dotenv>=1.0,<2.0" \
        "requests>=2.31,<3.0"

# Variables d'environnement par défaut (surchargeables via docker-compose)
ENV PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/opt/champy:/opt/airflow
