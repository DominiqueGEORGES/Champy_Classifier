# Régénération de l'historique authentique des modèles

> **Objectif** : reconstituer les 3 versions de modèles pour la soutenance
> (ResNet50 v1.0.0 baseline 84%, ResNet50 v1.1.0 aggressive 88%,
> ConvNeXt-Tiny v2.0.0 90%). Les 3 runs MLflow sur DagsHub sont déjà
> authentiques (du 30 mars au 23 avril 2026), mais les checkpoints
> ResNet50 ont été écrasés successivement sur le XPS pendant les
> itérations. Le ConvNeXt v2.0.0 est conservé sous
> `models/convnext_tiny_v2.0.0.{pt,onnx}` (préservé au Bloc R0).

## Pré-requis

- Le XPS 9520 doit être à jour (`git pull` sur `dev-dominique`).
- Le `.env` doit contenir `MLFLOW_TRACKING_USERNAME` et
  `MLFLOW_TRACKING_PASSWORD` (token DagsHub).
- DVC doit pouvoir tirer les données : `dvc pull`.

## Vendredi soir sur le XPS — ResNet50 default (~1h30)

```powershell
cd D:\DataScientest\Champy_Classifier
git pull origin dev-dominique
.venv\Scripts\Activate.ps1

# Vérifier l'env
type .env  # confirmer MLFLOW_TRACKING_USERNAME / PASSWORD

# Lancer le training (loggue dans MLflow DagsHub)
python -m src.training.train --config configs/training/default.yaml
```

À la fin, **noter le run_id MLflow** affiché en console (visible aussi
dans l'UI DagsHub). Il sera utilisé au moment de l'import dans le Model
Store BentoML pour la traçabilité.

### Préserver le checkpoint v1.0.0

```powershell
Copy-Item models\best_model.pt models\resnet50_v1.0.0.pt
python -m src.models.export_onnx `
    --checkpoint models\resnet50_v1.0.0.pt `
    --output models\resnet50_v1.0.0.onnx
Copy-Item models\class_names.json models\class_names_v1.0.0.json
```

L'export ONNX prend ~30s, valide les sorties numériques (max_diff < 1e-4
PyTorch vs ONNX).

## Vendredi soir suite — ResNet50 aggressive (~2h)

```powershell
python -m src.training.train --config configs/training/aggressive.yaml
```

Noter le nouveau run_id.

### Préserver le checkpoint v1.1.0

```powershell
Copy-Item models\best_model.pt models\resnet50_v1.1.0.pt
python -m src.models.export_onnx `
    --checkpoint models\resnet50_v1.1.0.pt `
    --output models\resnet50_v1.1.0.onnx
Copy-Item models\class_names.json models\class_names_v1.1.0.json
```

## Samedi matin — Transfert XPS -> NUC3

### Sur le XPS (source)

```powershell
cd D:\DataScientest\Champy_Classifier\models
python -m http.server 8888
```

Le port 8888 doit être autorisé sur le réseau local. Si problème de
firewall, autoriser python.exe en privé/public.

### Sur le NUC3 (destination)

Récupérer le hostname du XPS si besoin :

```powershell
# Depuis le XPS
hostname  # ex: Domi-XPS15-9520
```

Puis sur le NUC3 :

```powershell
cd D:\DataScientest\Champy_Classifier\models

# 6 fichiers à transférer (3 .onnx pour le serving + 3 .json pour les classes ;
# les .pt sont volumineux et restent sur le XPS, l'inférence se fait via ONNX)
$xps = "Domi-XPS15-9520"  # adapter au hostname réel
foreach ($f in @(
    "resnet50_v1.0.0.onnx", "resnet50_v1.1.0.onnx",
    "class_names_v1.0.0.json", "class_names_v1.1.0.json"
)) {
    Invoke-WebRequest "http://${xps}:8888/${f}" -OutFile $f
}
```

Ctrl+C sur le serveur HTTP du XPS une fois fini.

### Vérification d'intégrité

```powershell
# Sur le NUC3
Get-ChildItem models\*v1.* | Format-Table Name, Length, LastWriteTime
```

Tailles attendues :
- `resnet50_v1.0.0.onnx` ~90 MB
- `resnet50_v1.1.0.onnx` ~90 MB
- `class_names_*.json` ~750-800 B chacun

## Samedi midi — Import dans le Model Store BentoML (NUC3)

```powershell
cd D:\DataScientest\Champy_Classifier
.venv\Scripts\Activate.ps1

# v1.0.0 baseline (ResNet50 default)
python scripts\import_model_to_bentoml.py `
    --onnx-path models\resnet50_v1.0.0.onnx `
    --version v1.0.0 `
    --architecture resnet50 `
    --accuracy 0.84 `
    --class-names-path models\class_names_v1.0.0.json `
    --mlflow-run-id <run_id_v1.0.0>

# v1.1.0 aggressive (ResNet50 + lr++ / weight_decay)
python scripts\import_model_to_bentoml.py `
    --onnx-path models\resnet50_v1.1.0.onnx `
    --version v1.1.0 `
    --architecture resnet50 `
    --accuracy 0.88 `
    --class-names-path models\class_names_v1.1.0.json `
    --mlflow-run-id <run_id_v1.1.0>

# v2.0.0 (ConvNeXt-Tiny, deja preserve par le Bloc R0)
python scripts\import_model_to_bentoml.py `
    --onnx-path models\convnext_tiny_v2.0.0.onnx `
    --version v2.0.0 `
    --architecture convnext_tiny `
    --accuracy 0.90 `
    --class-names-path models\class_names_v2.0.0.json `
    --mlflow-run-id <run_id_v2.0.0>
```

### Vérification

```powershell
bentoml models list
# Doit afficher 3 tags champy_classifier:* avec ~106 MB chacun

# Inspecter les labels
python -c "
import bentoml
for tag in bentoml.models.list('champy_classifier'):
    m = bentoml.onnx.get(tag.tag)
    print(tag.tag, m.info.labels)
"
```

## Samedi après-midi — DVC versioning

L'objectif est de persister les 7 fichiers (`*_v1.0.0.*`, `*_v1.1.0.*`,
`*_v2.0.0.*`) dans DVC pour qu'ils soient récupérables sur n'importe
quelle machine via `dvc pull`.

```powershell
# Re-tracker tout models/ avec DVC (les fichiers .keras legacy sont aussi
# inclus, ce qui est OK : ils servent de référence historique)
dvc add models/
git add models.dvc .gitignore
git commit -m "feat: 3 model versions for MLOps demo (v1.0.0, v1.1.0, v2.0.0)"
git push origin dev-dominique
```

### Push DVC (sur ma confirmation explicite uniquement)

```powershell
dvc push
```

> **Attention** : ce push envoie ~600 MB sur le remote DagsHub
> (3 .pt + 3 .onnx + 3 .json + legacy .keras). Vérifier les quotas
> avant.

## Récapitulatif des artefacts attendus à la fin

```
models/
├── best_model.pt              # alias du modèle "courant" (= v2.0.0)
├── best_model.onnx            # alias ONNX courant
├── class_names.json           # alias classes courantes
├── resnet50_v1.0.0.pt
├── resnet50_v1.0.0.onnx
├── class_names_v1.0.0.json
├── resnet50_v1.1.0.pt
├── resnet50_v1.1.0.onnx
├── class_names_v1.1.0.json
├── convnext_tiny_v2.0.0.pt
├── convnext_tiny_v2.0.0.onnx
├── class_names_v2.0.0.json
├── cnn_tl_model.keras         # legacy archive
├── cnn_tl_model_history.npy   # legacy archive
├── cnn_tl2_model.keras        # legacy archive
└── cnn_tl2_model_history.npy  # legacy archive
```

Et dans le Model Store BentoML :

```
champy_classifier:<auto>  v1.0.0  resnet50      0.84
champy_classifier:<auto>  v1.1.0  resnet50      0.88
champy_classifier:<auto>  v2.0.0  convnext_tiny 0.90  <- latest
```

## Pièges connus

- **MLflow 401 silencieux** : si le `.env` est manquant ou que les env
  vars ne sont pas chargees dans la session PowerShell, MLflow renvoie
  des runs vides sans erreur claire. Verifier
  `$env:MLFLOW_TRACKING_PASSWORD` AVANT de lancer le train.
- **`best_model.pt` est ecrase a chaque run** : c'est pour ca qu'il faut
  IMPERATIVEMENT copier le checkpoint avec son nom versionne juste apres
  la fin du training. Sinon le run suivant ecrase tout.
- **L'export ONNX dynamo de torch 2.11+ peut produire un fichier vide
  (~240 KB au lieu de ~90 MB)** : `src/models/export_onnx.py` force
  `dynamo=False`, ne pas changer ce defaut.
- **Le `.onnx.data` orphelin** : si un export precedent a laisse un
  `models/best_model.onnx.data`, le supprimer avant le nouvel export
  (les modeles ConvNeXt-Tiny et ResNet50 sont self-contained, pas besoin
  de external data).
- **Transfert HTTP via `python -m http.server`** : si le XPS est en
  veille, le serveur ne repond pas. Desactiver la mise en veille pendant
  le transfert. Si timeout : relancer le serveur, le `Invoke-WebRequest`
  reprend a zero (pas de resume HTTP).
- **DVC pousse ~600 MB sur DagsHub** : verifier le quota du compte
  avant. Sinon on peut limiter en ne dvc-trackant que les `.onnx` (les
  `.pt` peuvent rester locaux puisque l'inference passe par ONNX).
