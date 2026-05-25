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

from invoke import task

PROJECT_ROOT = Path(__file__).parent


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
    c.run(f"python -m src.training.train --config {config}")


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
