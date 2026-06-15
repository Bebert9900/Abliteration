# AGENTS.md — piloter `abliteration` depuis un agent IA

Ce dépôt expose une CLI conçue pour être pilotée par un agent. Ce document est le guide
d'amorçage ; **la source de vérité machine est `python -m abliteration.cli schema --json`**
(liste exacte des commandes, arguments, types, défauts et formes de sortie, toujours à jour).

## Contrat de sortie (`--json`)

Toute sous-commande accepte le flag `--json`. Avec ce flag :

- **stdout** = une enveloppe unique versionnée, et rien d'autre :
  ```json
  {"schema_version": "1", "status": "ok", "command": "eval", "data": { ... }, "error": null}
  ```
  En cas d'échec : `"status": "error"`, `"data": null`,
  `"error": {"type": "<ExceptionType>", "message": "..."}`.
- **stderr** = tous les logs de progression/avertissements. Ne jamais parser stderr.

Sans `--json`, la sortie est destinée à un humain (texte/JSON simple) — ne pas s'y fier pour du parsing.

## Codes de sortie

| Code | Sens |
|------|------|
| `0`  | Succès |
| `1`  | Erreur d'exécution (exception attrapée ; détail dans `error` en mode `--json`) |
| `2`  | Erreur d'usage (argument invalide ; argparse écrit sur stderr) |

## Découverte

```bash
python -m abliteration.cli schema --json
```
Renvoie `{"version", "commands"}` où chaque commande décrit ses `arguments`
(`name`, `flags`, `type`, `default`, `required`, `choices`, `help`, `positional`) et la forme
de ses données de sortie (`output`).

## Commandes (résumé ; détails via `schema`)

| Commande | Rôle | `data` (succès) |
|----------|------|-----------------|
| `extract` | Collecte activations + directions 4 classes | `directions_path` |
| `select` | Sélection causale de la couche de refus | `selected_layer`, `scores` |
| `apply` | Orthogonalisation des poids + sauvegarde | `out_dir`, `selected_layer` |
| `abliterate` | Pipeline complet (extract→select→apply→eval) | métriques bi-axe + `out_dir` |
| `optimize` | Recherche Optuna (couche, alpha) co-minimisant refus + dégradations | `params`, `objective` |
| `eval` | Rapport bi-axe (refus + préservation), `--kl-map` optionnel | métriques + `kl_map` |
| `diagnose` | Séparabilité des directions (lecture seule) | `layers`, `warnings` |
| `concept-direction` | Direction d'un concept arbitraire (registre ou ad hoc) | `name`, `direction_path`, `norms_per_layer` |
| `concept-separability` | Matrice cosinus entre concepts | `concepts`, `matrix`, `warnings` |
| `concept-probe` | Décodabilité linéaire d'un concept couche par couche | `accuracy_per_layer`, `best_layer` |
| `concept-steer` | Pilotage causal par ajout de direction (génère avec/sans) | `comparisons`, `alpha`, `layer` |
| `analyze-circuit` | Localisation causale (refus par défaut, `--concept` pour tout concept) | `summary`, `report` |
| `heal` | Récupération agentique LoRA SFT | `out_dir` |
| `schema` | Auto-description machine de la CLI | `version`, `commands` |

## Exemple agent (bout en bout)

```bash
# 1) découvrir les capacités
python -m abliteration.cli schema --json

# 2) abliterer et récupérer les métriques de façon structurée
python -m abliteration.cli abliterate Qwen/Qwen2.5-3B-Instruct \
    --variant norm_preserving_biprojected --out ./artifacts/out --json

# 3) parser stdout : env["status"] == "ok" puis lire env["data"]["refusal_rate"], etc.
```

## Notes

- Les modules lourds (transformers, torch) sont chargés paresseusement : `schema` et la
  validation d'arguments fonctionnent sans GPU ni gros modèle.
- Conventions et garde-fous d'évaluation (holdout, `min_new_tokens`, juge LLM, deux axes
  refus/préservation) : voir le README, section « Évaluation ».
