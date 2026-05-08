"""Couche de serving BentoML pour le classificateur de champignons.

Ce package fournit une alternative a ``src/serving/`` (FastAPI). Le modele
ONNX servi est strictement identique : seule la couche de serving change
(adaptive batching, model store, packaging via bentofile.yaml).

Pieges BentoML 1.4 rencontres pendant la migration (a connaitre avant
de toucher au code) :

1. **``bentoml.onnx`` est deprecated depuis 1.4** : le sous-module reste
   fonctionnel mais sera retire dans une future version. Migration cible :
   ``bentoml.models.create()`` + chargement ONNX manuel via onnxruntime.

2. **``@bentoml.api`` force POST** : il n'y a pas de parametre ``method=``
   et tous les endpoints sont POST (style RPC). Pour des GET, il faudrait
   monter une ASGI app via ``@bentoml.asgi_app``. Choix retenu : on accepte
   POST pour ``/health`` et ``/model/info`` puisque le client (Streamlit
   et tests d'integration) maitrisent la methode.

3. **Appel intra-service async-only** : un endpoint qui en appelle un autre
   (par exemple ``predict`` -> ``infer_batch``) passe par un proxy RPC
   interne qui retourne une coroutine. La methode appelee DOIT etre
   ``async def`` et l'appel DOIT etre ``await self.infer_batch(...)``.

4. **Sérialisation float64 silencieuse via le proxy interne** : les
   tableaux numpy en transit entre endpoints sont promus en float64. ONNX
   Runtime exige float32 : caster avec
   ``np.ascontiguousarray(batch, dtype=np.float32)`` avant l'inference.

5. **``PIL.Image.Image`` doit rester un import runtime** : BentoML
   introspecte les annotations via ``typing.get_type_hints()`` au demarrage
   du worker pour brancher le decodeur d'image HTTP. Donc ``noqa: TC002``
   sur cet import (pas de ``TYPE_CHECKING`` block).

6. **``ModelOptions`` n'est pas un dict** : ``bento_model.info.options``
   ne supporte pas ``.get()``. Pour retrouver le fichier ONNX dans le
   Model Store, faire un glob sur ``saved_model.onnx`` (nom standard de
   ``bentoml.onnx.save_model``) avec fallback sur ``*.onnx``.

Validation parite FastAPI vs BentoML (Bloc 2 + add-on) : 4/4 images de
test, top-1 identique, delta max sur les confidences = 1.21e-07
(< epsilon 1e-6). Voir ``scripts/compare_fastapi_bentoml.py``.
"""
