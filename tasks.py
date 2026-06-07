"""
Task runner for Champy Classifier - replaces Makefile.
Cross-platform (Windows/Linux), uses invoke.

Usage (PowerShell):
    invoke setup
    invoke train
    invoke serve
    invoke --list        # voir toutes les commandes
"""

import shutil
from pathlib import Path

from invoke import Exit, task

PROJECT_ROOT = Path(__file__).parent

# Force le mode UTF-8 de Python (PEP 540) dans les sous-processus. Sans cela,
# sous Windows une console en cp1252 fait planter tout script qui ecrit un
# caractere hors cp1252 sur la sortie standard, par exemple l'emoji que MLflow
# affiche en fermant un run. On l'applique aux taches qui lancent un
# entrainement ou un deploiement.
UTF8_ENV = {"PYTHONUTF8": "1"}


@task
def setup(c):
    """Install dependencies and pull DVC data."""
    c.run("pip install -r requirements.txt")
    c.run("dvc pull")
    print("Setup complete.")


@task
def pull_data(c):
    """Pull data from DVC remote."""
    c.run("dvc pull")


@task
def split_data(c):
    """Run data split script."""
    c.run("python data/data_split.py")


@task
def train(c, config="configs/training/default.yaml"):
    """Launch training (native Python, for XPS with GPU)."""
    c.run(f"python -m src.training.train --config {config}", env=UTF8_ENV)


@task
def train_docker(c, config="configs/training/default.yaml"):
    """Launch training via Docker (requires NVIDIA Container Toolkit)."""
    c.run(
        f"docker compose -f docker-compose.train.yml run --rm --gpus all train "
        f"python -m src.training.train --config {config}"
    )


@task
def serve(c):
    """Start API + Demo + Monitoring (Docker Compose on NUC3)."""
    c.run("docker compose up -d api demo prometheus grafana")


@task
def serve_dev(c):
    """Start API + Demo in dev mode (with volume mounts)."""
    c.run("docker compose -f docker-compose.yml -f docker-compose.dev.yml up api demo")


@task
def stop(c):
    """Stop all Docker services."""
    c.run("docker compose down")


@task
def logs(c, service="api"):
    """Tail logs for a Docker service."""
    c.run(f"docker compose logs -f {service}")


@task
def test(c, verbose=True, cov=True, html=True):
    """Run tests with pytest + HTML report + coverage.

    Args:
        verbose: Affiche le détail de chaque test.
        cov: Mesure la couverture sur src/, demo/, monitoring/.
        html: Génère les rapports HTML dans reports/ (consommés par la page Streamlit CI/CD).
    """
    Path("reports").mkdir(exist_ok=True)

    cmd = "pytest tests/"
    if verbose:
        cmd += " -v"
    if html:
        cmd += " --html=reports/pytest.html --self-contained-html"
        cmd += " --junit-xml=reports/junit.xml"
    if cov:
        cmd += " --cov=src --cov=demo --cov=monitoring"
        cmd += " --cov-report=html:reports/coverage --cov-report=term"
    c.run(cmd)


@task
def test_unit(c):
    """Run unit tests only."""
    c.run("pytest tests/unit/ -v")


@task
def test_integration(c):
    """Run integration tests only."""
    c.run("pytest tests/integration/ -v")


@task
def lint(c):
    """Run linting (Ruff + Mypy)."""
    c.run("ruff check src/ tests/ demo/")
    c.run("ruff format --check src/ tests/ demo/")
    c.run("mypy src/")


@task
def format(c):
    """Format code with Ruff."""
    c.run("ruff check --fix src/ tests/ demo/")
    c.run("ruff format src/ tests/ demo/")


@task
def export_onnx(c):
    """Export model to ONNX format."""
    c.run("python -m src.models.export_onnx")


@task
def smoke(c, epochs=1):
    """Smoke test de la chaîne d'entraînement, rapide et de bout en bout.

    Dérive un profil court depuis configs/training/default.yaml (epochs réduits,
    pas de phase de backbone gelé), lance un entraînement, puis vérifie qu'une
    version 'champy-classifier' est bien apparue en Staging dans le registre
    MLflow. Ne mesure pas la performance : valide que la chaîne tourne, de
    l'entraînement jusqu'à l'enregistrement au registre. Réutilisable par tout
    repreneur du projet pour contrôler son installation.

    Args:
        epochs: Nombre d'epochs du smoke (1 par défaut).
    """
    # Imports différés : yaml et mlflow ne sont chargés que pour cette tâche,
    # pour ne pas alourdir chaque appel d'invoke (même logique que l'import
    # différé de bentoml dans scripts/import_model_to_bentoml.py).
    import yaml
    from mlflow import MlflowClient

    default_cfg = PROJECT_ROOT / "configs" / "training" / "default.yaml"
    smoke_cfg = PROJECT_ROOT / "configs" / "training" / "smoke.yaml"

    config = yaml.safe_load(default_cfg.read_text(encoding="utf-8"))
    config["total_epochs"] = epochs
    config["freeze_backbone_epochs"] = 0  # pas de phase 1 sur un smoke
    smoke_cfg.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Profil smoke écrit : {smoke_cfg} ({epochs} epoch(s))")

    c.run(f"python -m src.training.train --config {smoke_cfg}", env=UTF8_ENV)

    # Vérifie qu'une version vient bien d'atterrir en Staging dans le registre.
    versions = MlflowClient().get_latest_versions("champy-classifier", stages=["Staging"])
    if not versions:
        raise Exit("Aucune version 'champy-classifier' en Staging : la chaîne a échoué.", code=1)
    print(f"OK : champy-classifier v{versions[0].version} présent en Staging.")


@task
def deploy(c, stage="Staging"):
    """Déploie la version du registre MLflow vers le Model Store BentoML.

    Récupère la version du stage indiqué, l'exporte en ONNX et l'importe dans
    BentoML (cf. scripts/deploy_from_registry.py). À lancer sur le NUC3, là où
    vit le Model Store servi.

    Args:
        stage: Stage du registre à déployer (Staging par défaut, ou Production).
    """
    c.run(f"python -m scripts.deploy_from_registry --stage {stage}", env=UTF8_ENV)


@task
def build(c, no_cache=False):
    """Build all Docker images."""
    cmd = "docker compose build"
    if no_cache:
        cmd += " --no-cache"
    c.run(cmd)


@task
def clean(c):
    """Clean temporary files."""
    patterns = ["__pycache__", ".pytest_cache", ".mypy_cache", "htmlcov", ".ruff_cache"]
    for pattern in patterns:
        for path in PROJECT_ROOT.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed {path}")
    # Clean .pyc files
    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        pyc.unlink()
    print("Clean complete.")


@task
def status(c):
    """Show project status (Docker, DVC, Git)."""
    print("=== Docker services ===")
    c.run("docker compose ps", warn=True)
    print("\n=== DVC status ===")
    c.run("dvc status", warn=True)
    print("\n=== Git status ===")
    c.run("git status --short", warn=True)
