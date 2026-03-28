---
name: champy-mlops
description: >
  Skill MLOps pour le projet Champy Classifier (classification champignons, 30 espèces, 700K images).
  Environnement : Windows 11 Pro, PowerShell, Docker Desktop, pas de WSL.
  Utiliser pour toute tâche liée à : entraînement PyTorch, pipeline DVC, tracking MLflow (DagsHub),
  serving FastAPI, export ONNX, monitoring Prometheus/Grafana, drift detection Evidently,
  Dockerisation des services, CI/CD GitHub Actions, demo Streamlit.
  Aussi utiliser quand le user mentionne : mushroom, champignon, ResNet, training, inference,
  model registry, data pipeline, batch size, VRAM, mixed precision, GradCAM.
---

# Champy MLOps Skill

## Règle de documentation obligatoire

**A chaque fin d'étape ou de bloc de travail**, mettre à jour le fichier `LOGBOOK.md` à la racine du projet :
1. Remplir la section correspondante (date, décisions, problèmes, artefacts, métriques)
2. Le tableau "Décisions prises" doit TOUJOURS contenir : le choix fait, les alternatives envisagées, et la justification
3. Les problèmes rencontrés et leurs solutions doivent être documentés - y compris les erreurs et les culs-de-sac
4. Les métriques et résultats chiffrés sont obligatoires quand ils existent
5. **Demander confirmation au user** avant de considérer une étape comme terminée

Ce logbook sert au mémoire de Master. La qualité de la documentation compte autant que le code.

**En parallèle**, enrichir le fichier `PLAYBOOK.md` à la racine du projet :
- Ajouter les **pièges connus** rencontrés à chaque étape (erreurs, incompatibilités, surprises)
- Compléter les **commandes clés** effectivement utilisées
- Ajuster les **durées typiques** en fonction du vécu
- Le PLAYBOOK doit rester **générique et réutilisable** : pas de détails spécifiques à Champy, mais des leçons transposables à tout projet MLOps

## Contexte projet

- Classification 30 espèces de champignons, ~700K images
- Modèle : ResNet50 fine-tuned (transfer learning ImageNet)
- **Environnement : Windows 11 Pro, PowerShell, Docker Desktop (pas de WSL)**
- Contrainte GPU : RTX 3050 Ti (4GB VRAM) sur XPS, training natif Windows (pas Docker GPU)
- Hub MLOps : NUC3 (Ryzen AI 9, 96GB RAM, CPU only, Docker Desktop) - serving, monitoring, dev
- Remote DVC + MLflow : DagsHub (LoicFocraud/Champy_Classifier)
- Tout est dockerisé (sauf le training qui tourne nativement sur XPS)

## Contraintes Windows - Rappels

- **Pas de bash** : toutes les commandes en PowerShell ou via Python
- **Chemins** : `pathlib.Path` exclusivement, jamais de `/` ou `\` hardcodé
- **Line endings** : LF dans le repo (`.gitattributes`)
- **Task runner** : `invoke` (tasks.py), pas Make
- **Docker volumes** : syntaxe Windows (`${PWD}` fonctionne en PowerShell)
- **Pas de `rm -rf`** : utiliser `Remove-Item -Recurse -Force` ou `shutil.rmtree()` en Python
- **Variables d'environnement** : `$env:VAR` en PowerShell, pas `$VAR` ou `export VAR=`

## Patterns récurrents

### Entraînement avec contrainte VRAM (XPS, natif Windows)

```python
# Toujours utiliser mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- Batch size : commencer à 16, monter si la VRAM le permet
- Si OOM : gradient accumulation (accumulate_steps=2 ou 4)
- `torch.backends.cudnn.benchmark = True` pour convolutions fixes
- Libérer le cache : `torch.cuda.empty_cache()` entre les epochs si nécessaire
- **DataLoader** : `num_workers=0` par défaut sur Windows (multiprocessing fork pas supporté), tester `num_workers=2` avec `persistent_workers=True`

### MLflow logging standard

```python
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/LoicFocraud/Champy_Classifier.mlflow")

with mlflow.start_run(run_name="resnet50_v2"):
    mlflow.log_params({
        "model": "resnet50",
        "lr": lr,
        "batch_size": batch_size,
        "epochs": max_epochs,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "seed": seed,
        "mixed_precision": True,
    })
    mlflow.log_metrics({"train_loss": t_loss, "val_loss": v_loss, "val_acc": v_acc}, step=epoch)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.pytorch.log_model(model, "model")
```

### FastAPI serving pattern

```python
from fastapi import FastAPI, UploadFile
import onnxruntime as ort
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(title="Champy Classifier API", version="1.0.0")

PREDICTIONS = Counter("predictions_total", "Total predictions", ["species"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

session = ort.InferenceSession("model.onnx")

@app.post("/predict")
async def predict(file: UploadFile):
    with LATENCY.time():
        image = preprocess(await file.read())
        outputs = session.run(None, {"input": image})
        top5 = decode_top5(outputs)
    PREDICTIONS.labels(species=top5[0]["species"]).inc()
    return {"predictions": top5}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}
```

### Export ONNX

```python
dummy = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
import onnx
model_onnx = onnx.load("model.onnx")
onnx.checker.check_model(model_onnx)
```

### Docker Compose pattern (NUC3, Docker Desktop)

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    env_file: .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      retries: 3

  demo:
    build:
      context: .
      dockerfile: docker/Dockerfile.demo
    ports:
      - "8501:8501"
    env_file: .env
    depends_on:
      api:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./configs/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  grafana-data:
```

### Streamlit - Pattern zéro hardcoded

Le Streamlit est un portfolio MLOps interactif. Il ne stocke rien, il lit tout dynamiquement.
Il ne remplace ni MLflow (carnet de labo) ni Grafana (salle de contrôle). C'est la vitrine pour jury/clients.

**Helpers partagés** dans `demo/lib/` :

```python
# demo/lib/mlflow_utils.py
import mlflow
import streamlit as st
from src.config import settings

@st.cache_data(ttl=60)
def get_best_run(metric: str = "val_acc") -> dict:
    """Récupère le meilleur run depuis MLflow. Jamais de valeur hardcodée."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        return {}
    return runs.iloc[0].to_dict()

@st.cache_data(ttl=60)
def get_model_versions(name: str = "champy-classifier") -> list:
    """Liste les versions du modèle depuis le registry."""
    client = mlflow.tracking.MlflowClient()
    return client.search_model_versions(f"name='{name}'")
```

```python
# demo/lib/data_utils.py
from pathlib import Path
import streamlit as st

@st.cache_data
def scan_dataset(data_dir: Path) -> dict:
    """Scanne le répertoire et retourne les stats par classe. Zéro hardcoded."""
    stats = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            stats[class_dir.name] = len(images)
    return stats
```

**Pattern standard pour chaque page** :
```python
# Page Streamlit - toujours avec fallback explicite
try:
    best_run = get_best_run()
    st.metric("Best accuracy", f"{best_run['metrics.val_acc']:.1%}")
except Exception as e:
    st.warning(f"Source non disponible : {e}")
```

**Construction incrémentale** : les pages se créent au fil des étapes. Etape data terminée -> pages 01-04. Training terminé -> pages 05-06. Etc.

### GradCAM pour explicabilité

```python
import torch.nn.functional as F

def gradcam(model, image_tensor, target_class):
    model.eval()
    features = {}
    def hook(module, input, output):
        features["last_conv"] = output
    handle = model.layer4[-1].register_forward_hook(hook)

    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    grads = features["last_conv"].grad
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * features["last_conv"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear")
    cam = cam / cam.max()

    handle.remove()
    return cam.squeeze().detach().cpu().numpy()
```

### Evidently drift report

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
report.run(reference_data=ref_df, current_data=curr_df)
report.save_html("drift_report.html")
```

## Checklist de livraison MLOps

- [ ] Data pipeline reproductible (DVC + split script)
- [ ] Training pipeline avec MLflow tracking
- [ ] Model registry (MLflow, staging/production)
- [ ] Export ONNX validé
- [ ] API FastAPI avec /predict, /health, /metrics
- [ ] Tests unitaires + intégration (>80% coverage)
- [ ] Streamlit portfolio 13 pages (zéro hardcoded, sources dynamiques)
- [ ] Docker Compose (api + demo + prometheus + grafana)
- [ ] CI/CD GitHub Actions (lint + test + build)
- [ ] Monitoring (latence, throughput, distribution classes)
- [ ] Drift detection (Evidently)
- [ ] Documentation (README complet)
- [ ] GradCAM explicabilité
- [ ] tasks.py avec toutes les commandes
- [ ] LOGBOOK.md complété pour chaque étape
- [ ] PLAYBOOK.md enrichi (pièges, commandes, durées)