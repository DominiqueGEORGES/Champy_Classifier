"""DAG de reentrainement Champy : la boucle complete entrainement -> deploiement.

Orchestre la chaine MLOps de bout en bout, par SSH, depuis Airflow (qui tourne
en conteneur sur le NUC3) :

1. entrainement_xps2 : se connecte au XPS2 (la machine GPU), met le code a jour
   (git pull) puis lance l'entrainement. Le modele est publie au registre MLflow
   du NUC3 et place en Staging.
2. deploiement_nuc3 : se connecte au NUC3, recupere la version Staging du
   registre, l'exporte en ONNX et l'importe dans le Model Store BentoML.

Declenchement manuel (schedule=None). Le parametre `training_command` choisit la
tache Invoke d'entrainement : 'smoke' (entrainement court de validation, defaut)
ou 'train' (entrainement complet). Voir les prerequis en bas du fichier.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.ssh.operators.ssh import SSHOperator

# Commandes en PowerShell pur. Le shell SSH par defaut des deux machines doit
# etre PowerShell (voir prerequis) : ainsi pas de wrapper `powershell -Command`
# ni de guillemets imbriques. invoke porte deja PYTHONUTF8=1 (cf. tasks.py),
# ce qui evite le crash d'encodage des emojis MLflow en session non interactive.
ENTRAINEMENT_CMD = (
    "$env:PYTHONUTF8='1'; "
    "Set-Location 'D:\\projets\\DataScientest\\Champy_Classifier'; "
    "$env:MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING='true'; "
    "git pull; if ($LASTEXITCODE -ne 0) { throw 'echec git pull' }; "
    ".\\.venv\\Scripts\\invoke.exe {{ params.training_command }}"
)

DEPLOIEMENT_CMD = (
    "$env:PYTHONUTF8='1'; "
    "Set-Location 'D:\\DataScientest\\Champy_Classifier'; "
    ".\\.venv\\Scripts\\invoke.exe deploy"
)

with DAG(
    dag_id="champy_reentrainement",
    dag_display_name="02_champy_reentrainement",
    description="Boucle MLOps Champy : entrainement (XPS2) puis deploiement (NUC3), par SSH.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["champy", "mlops"],
    params={
        "training_command": Param(
            "smoke",
            type="string",
            enum=[
                "smoke",
                "train --config configs/training/convnext.yaml",
            ],
            description="Tache Invoke sur le XPS2 : smoke (court) ou train complet (ConvNeXt-Tiny).",
        ),
    },
) as dag:
    entrainement = SSHOperator(
        task_id="entrainement_xps2",
        ssh_conn_id="xps2_ssh",
        command=ENTRAINEMENT_CMD,
        conn_timeout=30,
        cmd_timeout=4 * 60 * 60,  # jusqu'a 4 h, pour couvrir un entrainement complet
    )

    deploiement = SSHOperator(
        task_id="deploiement_nuc3",
        ssh_conn_id="nuc3_ssh",
        command=DEPLOIEMENT_CMD,
        conn_timeout=30,
        cmd_timeout=30 * 60,  # 30 min : download + export ONNX + import BentoML
    )

    entrainement >> deploiement


# ---------------------------------------------------------------------------
# PREREQUIS (a faire une seule fois)
# ---------------------------------------------------------------------------
# 1. Provider SSH dans l'image Airflow :
#       pip install apache-airflow-providers-ssh
#
# 2. OpenSSH Server actif sur le XPS2 ET le NUC3, shell par defaut = PowerShell.
#    Sur chaque machine (PowerShell admin) :
#       Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
#       Start-Service sshd
#       Set-Service -Name sshd -StartupType Automatic
#       New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
#         -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
#         -PropertyType String -Force
#    (Changer le DefaultShell affecte toutes les sessions SSH entrantes.)
#
# 3. Authentification par cle (pas par mot de passe) : la cle publique d'Airflow
#    dans C:\Users\<user>\.ssh\authorized_keys des deux machines.
#
# 4. Deux connexions Airflow (Admin > Connections), type SSH :
#       - xps2_ssh : host = <IP Tailscale du XPS2>, login = <user>, cle privee
#       - nuc3_ssh : host = 100.64.59.51 (ou host.docker.internal selon le reseau
#                    du conteneur), login = <user>, cle privee
#
# 5. Le conteneur Airflow doit pouvoir router vers le tailnet, sinon il ne joint
#    ni le XPS2 ni le NUC3. A verifier au premier run : reseau du conteneur
#    (network_mode host, Tailscale joignable depuis le conteneur, ou
#    host.docker.internal pour atteindre le NUC3).
