# Pre-commit : la boucle de reformatage qui fait échouer le premier commit

Date : 8 juin 2026
Sujet : pourquoi un `git commit` échoue régulièrement « tout seul » sur ce projet, et comment l'éviter.

---

## Le symptôme

À chaque commit d'un fichier fraîchement déposé, le commit échoue une première fois. La sortie ressemble à :

```
fix end of files........................................Failed
- files were modified by this hook
Fixing tasks.py
ruff (legacy alias).....................................Failed
- files were modified by this hook
Found 1 error (1 fixed, 0 remaining).
```

Aucun commit n'est créé. Si on enchaîne un `git push` derrière, il répond « Everything up-to-date » puisqu'il n'y a rien de nouveau à envoyer.

---

## La cause

Le projet utilise **pre-commit**, un mécanisme qui lance des vérifications automatiques juste avant chaque commit. Certaines de ces vérifications ne se contentent pas de vérifier : elles **corrigent** le fichier (on parle de hooks « auto-fix »). Ici, deux d'entre elles :
- `end-of-file-fixer` : normalise la fin de fichier (une seule ligne vide finale, fins de ligne unifiées).
- `ruff` : corrige automatiquement certains détails de style.

La règle de pre-commit est simple : **si un hook modifie un fichier pendant le commit, le commit est annulé**, pour que tu valides toi-même la version corrigée.

Le piège vient de l'ordre. Quand tu fais `git add` puis `git commit` :
1. `git add` met en zone de validation la version actuelle du fichier.
2. `git commit` déclenche les hooks, qui corrigent le fichier **après** le `git add`.
3. La version corrigée n'étant pas dans la zone de validation, pre-commit annule le commit.

Le fichier sur le disque est alors corrigé, mais ce qui était mis en validation ne l'est pas : d'où l'annulation.

Pourquoi ça revient si souvent ? Les fichiers livrés depuis un autre environnement arrivent avec des fins de ligne de style Unix (LF), que Windows et la configuration git du projet normalisent. La première rencontre avec les hooks déclenche donc presque toujours une correction.

---

## La parade

Laisser les hooks corriger **avant** le `git add`. Le commit passe alors du premier coup :

```powershell
pre-commit run --files tasks.py   # les hooks corrigent le fichier sur le disque (un "Failed" ici est normal et attendu)
git add tasks.py                  # on met en validation la version deja corrigee
git commit -m "..."               # plus rien a corriger, le commit aboutit
```

À défaut, la méthode « brute » fonctionne aussi : refaire `git add` puis `git commit` après l'échec. Comme le fichier est désormais corrigé sur le disque, le second essai aboutit. C'est juste moins propre, et ça affiche un échec inutile à chaque fois.

---

## À retenir

- Un commit « qui échoue » à cause de `end-of-file-fixer` ou `ruff` n'est pas une erreur de code : c'est un reformatage. Le seul vrai blocage serait une erreur non auto-corrigeable (par exemple un `ruff` qui signale `0 fixed, 1 remaining`).
- Toujours vérifier la présence de la ligne `[dev-dominique xxxxxxx] ...` après `git commit` : c'est la preuve qu'un commit a réellement été créé, avant de faire `git push`.
- Pour limiter le problème à la source, les fichiers Python peuvent être passés à `ruff format` puis `ruff check --fix` avant d'être déposés ; reste la normalisation des fins de ligne, qui dépend du transfert vers Windows.
