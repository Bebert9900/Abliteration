# Meridian

**Cartographier et surveiller les directions qu'un LLM encode.** Meridian extrait, suit et
modifie les directions de concepts dans les activations d'un modèle. Son usage premier : **suivre
l'évolution des représentations internes au fil d'un fine-tuning** — pour détecter tôt qu'une
capacité dérive, s'effondre ou s'intrique avec une autre. L'**ablation dirigée** (« abliteration »,
directional ablation) en est l'une des capacités, pas le cœur.

Tout repose sur le même objet géométrique : une **direction** = un contraste de moyennes
d'activations entre deux ensembles de prompts, `d̂ = normalize(μ_pos − μ_neg)`, calculée couche
par couche. À partir de là, Meridian sait : construire un **atlas** de directions (par sujet et
par variance), **identifier** la direction d'un sujet arbitraire, **suivre** la dérive de ces
directions dans le temps, et **abliter** une direction de refus en préservant les capacités.

## Surveiller un fine-tuning (cas d'usage premier)

Un fine-tuning modifie les poids ; les directions internes bougent avec eux, souvent sans qu'on
le voie dans la loss : une direction de sujet **tourne** (la représentation se réorganise), sa
**force** s'effondre (le modèle « oublie » un sujet), deux concepts s'**intriquent**, ou le
**sous-espace latent** dérive. Meridian rend ces phénomènes mesurables, checkpoint par checkpoint
ou en direct pendant l'entraînement.

**En ligne** — brancher un callback sur un `transformers.Trainer` ; il construit un atlas à chaque
sauvegarde et écrit la série de dérive (référence = premier instantané) :

```python
from meridian.atlas import AtlasDriftCallback, load_labeled
from meridian.data import PromptFormatter

cb = AtlasDriftCallback(load_labeled("data/labeled_demo.jsonl"),
                        PromptFormatter(tokenizer), k=32, out_path="results/atlas_drift.json")
trainer.add_callback(cb)
```

**Hors-ligne** — sur des checkpoints déjà sauvegardés (trainer-agnostique) :

```bash
python -m meridian.cli atlas-monitor \
    --checkpoints ckpt-100,ckpt-200,ckpt-final \
    --dataset data/labeled_demo.jsonl --report results/atlas_monitor.json
```

Métriques de dérive (résumé agrégé sur les couches milieu, où les directions sont stables) :

| Mesure | Métrique | Plage |
|---|---|---|
| Dérive d'une direction de sujet | distance cosinus **signée** `1 − cos` (direction orientée : un flip de signe compte) | [0, 2] |
| Dérive du sous-espace latent | angles principaux / distance de Grassmann (signe SVD non orienté) | [0, 1] |
| Force d'un sujet | variation de `‖μ_s − μ_reste‖` (renforcement / effondrement) | — |
| Intrication | évolution de la matrice cosinus inter-sujets | — |

> Le suivi n'est valide qu'au sein d'une **même lignée** (même architecture/base → base de hidden
> states partagée). Comparer des modèles non apparentés n'a pas de sens géométrique.

## Atlas de directions

À partir d'un **dataset étiqueté** (textes taggés par sujet — un JSONL `{"text": ..., "subject": ...}`
ou un dossier d'un `.txt` par sujet), `atlas-build` calcule en une passe :

- une **direction supervisée par sujet** : `d̂_s = normalize(μ_s − μ_reste)` (one-vs-rest) ;
- un jeu de **directions latentes non supervisées** : SVD/PCA par couche, avec variance expliquée ;
- le **pont** entre les deux (quelle latente ≈ quel sujet, et inversement) et la séparabilité
  inter-sujets.

L'atlas est sérialisé en `.safetensors` (directions seulement, jamais de poids de modèle).

```bash
python -m meridian.cli atlas-build   <model> --dataset data/labeled_demo.jsonl --k 32 \
    --out results/atlas.safetensors
python -m meridian.cli atlas-identify --atlas results/atlas.safetensors --subject biologie
```

`atlas-identify` répond à « quelle est la direction de ce sujet, et qu'est-ce qui lui ressemble ? » :
il renvoie les sujets les plus proches d'une direction (par `|cos|` — recherche d'axe), à partir
d'un sujet de l'atlas (`--subject`) ou d'une direction `.pt` quelconque (`--direction`). Entièrement
hors-ligne (aucun chargement de modèle).

## Concepts

L'abstraction `concepts/` généralise la notion de direction à tout concept défini par un contraste
de prompts. Trois concepts sont prédéfinis (`refusal`, `negation`, `agentic`) ; on peut en charger
un depuis deux fichiers. Lecture seule.

```bash
python -m meridian.cli concept-direction    <model> --concept refusal
python -m meridian.cli concept-separability  <model> --concepts refusal,negation,agentic
python -m meridian.cli concept-probe         <model> --concept refusal
python -m meridian.cli concept-steer         <model> --concept refusal --alpha 8
```

## Ablation dirigée (abliteration)

L'une des capacités de Meridian : retirer la **direction de refus** d'un modèle sans
réentraînement (directional ablation, Arditi et al. 2024), en cherchant à préserver deux capacités
qu'une ablation naïve dégrade — la **négation logique** (« non, ce code est faux ») et l'**agentique**
(appels d'outils, sorties structurées). C'est l'enjeu de l'« abliteration préservante ».

### Principe

On calcule la direction de refus par contraste de moyennes d'activations (prompts nuisibles vs
neutres), `r = normalize(μ_harmful − μ_harmless)`. Plutôt que d'effacer `r` directement, on
l'orthogonalise d'abord contre les directions à préserver (négation `n`, agentique `a`), de sorte
que l'ablation ne touche que la composante du refus indépendante de ces capacités :

```
r_safe = project_out(r, [n, a, ...])
W'     = W − r_safe (r_safeᵀ W)
```

L'orthogonalisation s'applique à toutes les matrices écrivant dans le residual stream (`o_proj`,
`down_proj` de chaque couche, embeddings ; un `down_proj` par expert pour les MoE).

### Les quatre classes de prompts

| Classe | Rôle |
|---|---|
| `harmful` | Déclenche le refus ; combinée à `harmless`, donne la direction de refus |
| `harmless` | Référence neutre |
| `legitimate_negation` | Négation logique légitime, à préserver |
| `agentic` | Appels d'outils et sorties structurées, à préserver |

### Variantes (`--variant`)

| Variante | Description |
|---|---|
| `conventional` | `W' = W − r(rᵀW)` ; efface la direction brute (baseline) |
| `projected` | Orthogonalise `r` contre `harmless` avant ablation |
| `preserving` | Orthogonalise `r` contre un sous-ensemble choisi de directions |
| `norm_preserving_biprojected` | Idem `preserving`, avec préservation de la norme des poids (défaut de prod) |

### Pipeline

```bash
# Pipeline complet : extract → select → apply → eval
python -m meridian.cli abliterate meta-llama/Llama-3.1-8B-Instruct \
    --variant preserving --preserve negation,agentic --data-dir data --out ./out

# Étapes séparées
python -m meridian.cli extract <model> --data-dir data --out directions.pt
python -m meridian.cli select  <model> --directions directions.pt
python -m meridian.cli apply   <model> --directions directions.pt \
    --variant preserving --preserve negation,agentic --layer 14 --out ./out

# Diagnostic (lecture seule) de la séparabilité des directions
python -m meridian.cli diagnose <model> --directions directions.pt --layers 8-20
```

La commande `apply`/`abliterate` sauvegarde le modèle en safetensors et une model card (modèle de
base, méthode, directions préservées, métriques).

### Optimisation

```bash
python -m meridian.cli optimize <model> --trials 50 \
    --lambda-kl 1.0 --lambda-neg 2.0 --lambda-syco 0.5 --lambda-agent 3.0
```

L'objectif Optuna co-minimise le refus et les dégradations de préservation :

```
objectif = refusal_rate + λ_kl·KL(harmless) + λ_neg·(1 − negation_retention)
         + λ_syco·follow_rate + λ_agent·(1 − agentic_score)
```

Le terme `λ_agent` évite qu'un optimum sur (refus, KL) masque un effondrement des appels d'outils.
La boucle persiste ses essais en JSONL et reprend après interruption.

### Évaluation (bi-axe) et récupération

```bash
python -m meridian.cli eval ./out --benchmarks mmlu,gsm8k --out report.json
python -m meridian.cli heal ./out --traces traces.jsonl --n-traces 200   # extra `heal` (peft)
```

Le rapport `eval` couvre **deux axes** : refus (`refusal_rate` sur le holdout + filtre de
dégénérescence) et préservation (`kl`, `negation_retention`, `agentic_score`), avec garde-fous
(`degeneracy_rate`, `empty_rate`, `follow_rate`). Optimiser le refus seul produit un modèle
dé-censuré mais lobotomisé : les deux axes sont non négociables.

Le juge de refus par défaut est heuristique (mots-clés, déterministe, auditable). Il manque les
refus déguisés ; un juge LLM hors-ligne (`meridian/eval/llm_judge.py`) permet de reclasser après
coup des sorties déjà générées (`REFUSAL` / `NON_REFUSAL` / `EVASIVE`), après validation contre des
labels humains. `heal` est une récupération optionnelle : si l'agentique s'est effondrée, un court
LoRA SFT sur ~100–300 traces d'appels d'outils la restaure sans réintroduire le refus.

## Architecture

Le code ne dépend d'aucun nom de module codé en dur : `ArchAdapter` localise les matrices écrivant
dans le residual stream par introspection (`named_modules()`, matching par suffixe). Sont gérées
les architectures denses, les MoE (un `down_proj` par expert + couches partagées) et le Conv1D de
type GPT-2 (axes transposés).

```
meridian/             Package Python
├── cli.py            Point d'entrée : python -m meridian.cli <commande>
├── atlas/            Atlas de directions : sujets (supervisé), latents (SVD), pont, dérive, callback
├── concepts/         Abstraction « concept » générique (direction, séparabilité, sonde)
├── data/             Classes contrastives, chat template, holdout déterministe
├── models/           Chargement bf16 et ArchAdapter
├── directions/       Collecte d'activations, calcul des directions, sélection de couche
├── ablation/         project_out, variantes, orthogonalisation des poids, hooks réversibles
├── eval/             Refus, KL, négation, agentique, benchmarks, juge LLM hors-ligne
├── circuits/         Analyse circuitielle (DLA + patching causal), lecture seule
├── optimize/         Objectif composite et boucle Optuna
├── io/               Sauvegarde safetensors et model card
├── cache.py          Cache disque des calculs déterministes
├── output.py         Contrat de sortie --json
└── heal.py           Récupération agentique par LoRA SFT

scripts/   Expérimentation et reproduction   tests/   Suite pytest (miroir du package)
data/      Prompts (JSONL)                    results/ Rapports de mesure (scores agrégés)
```

## Installation

Python ≥ 3.10, PyTorch ≥ 2.2. La gestion des dépendances cible `uv`, mais `pip` fonctionne.

```bash
uv sync                  # ou : pip install -e .

pip install -e ".[optimize]"   # optuna
pip install -e ".[eval]"       # lm-eval (MMLU, GSM8K, ...)
pip install -e ".[heal]"       # peft (LoRA SFT)
pip install -e ".[quant]"      # bitsandbytes (mesure uniquement)
pip install -e ".[dev]"        # pytest, ruff
```

La quantification 4-bit n'est utilisée que pour mesurer de gros modèles ; les poids ablitérés sont
toujours produits en bf16.

## Format des données

Un fichier JSONL par classe (extension `.txt`, contenu JSONL). Chaque ligne porte une clé `text`
(ou `prompt`) ; toute autre clé est conservée dans `meta` (utile pour `agentic`). Pour l'atlas, un
dataset étiqueté est soit un JSONL unique avec une clé de label (`--label-key`, défaut `subject`),
soit un dossier d'un `.txt` par sujet.

```
data/
├── harmful.txt   harmless.txt   legitimate_negation.txt   agentic.txt   # 4 classes (ablation)
└── labeled_demo.jsonl                                                   # dataset étiqueté (atlas)
```

Chaque ensemble est découpé en train/holdout de façon déterministe (graine fixe) : le train
calcule les directions, le holdout mesure sur des prompts non vus.

## Pilotage par des agents IA

La CLI est conçue pour être pilotée par un programme ou un agent, pas seulement par un humain.
Chaque sous-commande accepte `--json` : stdout ne contient qu'une enveloppe versionnée, les logs
vont sur stderr.

```json
{"schema_version": "1", "status": "ok", "command": "atlas-monitor", "data": { ... }, "error": null}
```

En cas d'échec, `status` vaut `"error"`, `data` est `null`, `error` porte `{type, message}`. Codes
de sortie : `0` succès, `1` erreur d'exécution, `2` erreur d'usage. `schema --json` décrit toute la
CLI (commandes, arguments, types, défauts, formes de sortie) sans charger torch ni modèle.

```bash
python -m meridian.cli schema --json
python -m meridian.cli atlas-build Qwen/Qwen2.5-3B-Instruct --dataset data/labeled_demo.jsonl \
    --out results/atlas.safetensors --json
```

Guide d'intégration détaillé dans `AGENTS.md` ; la source de vérité reste `schema --json`.

## État d'implémentation

| Composant | État |
|---|---|
| `atlas` (sujets, latents SVD, pont, dérive, `AtlasDriftCallback`) | Implémenté et testé |
| `concepts`, `cache`, contrat `--json` | Implémenté et testé |
| `data`, `models`, `directions`, `ablation` | Implémenté et testé |
| `eval` (refus, KL, négation, agentique, juge hors-ligne) | Implémenté et testé |
| `optimize` (objectif composite, boucle Optuna) | Implémenté et testé |
| `heal` (LoRA SFT) | Implémenté ; nécessite l'extra `heal` |
| `io` (safetensors, model card) | Implémenté et testé |
| `circuits` (Phase 1, lecture seule) | Implémenté et testé |
| CLI (17 sous-commandes) | Câblée aux modules réels |
| `io.export_gguf` | Non implémenté (nécessite llama.cpp) |
| Analyse circuitielle Phase 2 (ablation ciblée) | Non implémenté |

### Résultats mesurés (ablation)

Qwen2.5-3B-Instruct, variante `preserving`, couche 22, holdout de 30 prompts nuisibles :

| Métrique | Base | Ablitéré |
|---|---|---|
| `refusal_rate` (heuristique) | 0.90 | 0.00 |
| `refusal_rate` (lecture humaine) | — | ≈ 0.07 |
| `negation_retention` | 0.93 | 0.96 |
| `agentic_score` | 0.97 | 1.00 |
| MMLU (limit 30) | — | 0.64 |
| GSM8K (limit 50) | — | 0.46 |
| KL de préservation | — | 0.78 |

Le `refusal_rate` heuristique de 0.00 est optimiste : la lecture humaine des 30 réponses montre
environ 90 % de compliances réelles et deux refus déguisés non détectés par les mots-clés, soit un
taux de refus effectif autour de 7 %.

## Tests

```bash
pytest                  # suite complète (~265 tests)
pytest -m "not model"   # exclut les tests qui chargent un modèle
ruff check .            # lint (gate CI)
```

Les fonctions pures (directions, SVD, dérive, séparabilité, atlas) sont testées sur des tenseurs
jouets, sans modèle. Le test d'intégration `tests/integration/test_constraints.py` utilise un
modèle jouet à matrices d'écriture identité, qui rend la rétention d'une sonde calculable
exactement (`rétention = 1 − (d·p)²`), et vérifie sur la chaîne réelle que `preserving` conserve la
négation et l'agentique là où `conventional` les dégrade.

### Validation du juge

Le `refusal_rate` ne vaut que ce que vaut le juge. Les sorties ont été rejugées hors-ligne puis le
juge comparé à des labels humains : le juge LLM 3B local n'a pas passé la validation (biais de
nocivité), la référence reste donc le label humain, qui confirme que l'ablation est solide. Deux
règles : valider tout juge automatique sur un échantillon humain, et conserver les textes bruts des
générations (pas seulement les scores) pour pouvoir rejuger. Ces textes peuvent contenir du contenu
nuisible et restent hors du dépôt.

## Points d'attention

- Appliquer le chat template avant toute collecte d'activations, sinon la direction est bruitée.
- Padding à gauche (ou indexation par `attention_mask`) pour capter le dernier token.
- Orthogonaliser toutes les écritures résiduelles (`o_proj`, `down_proj`, embeddings ; chaque
  expert pour les MoE) ; en oublier laisse du refus résiduel.
- Produire les poids en bf16 ; la 4-bit est réservée à la mesure.
- Passer par `ArchAdapter` plutôt que de coder des noms de modules en dur.
- Hooks réversibles (`meridian/ablation/hooks.py`) pour explorer/sélectionner une couche ;
  orthogonalisation permanente seulement à la livraison.

## Analyse circuitielle (Phase 1)

`meridian/circuits/` cherche quels composants (têtes d'attention, MLP) portent le refus et comment
l'information circule. Phase 1 = lecture seule (aucune modification de poids). Règle d'or : la DLA,
corrélationnelle, ne conclut jamais seule ; toute localisation est confirmée par patching causal
(nécessité et suffisance).

```bash
python -m meridian.cli analyze-circuit Qwen/Qwen3-0.6B --device cuda \
    --pairs 16 --top-k 24 --threshold 0.5 --n-boot 300 --out rapport.json
```

Le backend par défaut est `TorchHookBackend` ; NNsight est un backend de lecture alternatif. La
Phase 2 (ablation ciblée) n'est pas implémentée : sur Qwen3-0.6B la localisation s'est révélée
instable d'un run à l'autre (Jaccard bootstrap < 0.9).

## Cadre d'usage

Meridian est un outil de recherche en interprétabilité, à des fins éducatives et défensives.
L'ablation dirigée est une technique à double usage : retirer la direction de refus lève des
garde-fous. Le dépôt reste un **outil générique de modification/observation de modèle** :

- Toute livraison de poids s'accompagne d'une model card (modèle de base, méthode, métriques).
- L'évaluation bi-axe fait partie intégrante du pipeline d'ablation.
- Aucune fonctionnalité n'est orientée vers la production de contenu gravement dangereux.

## Référence

Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.

## Disclaimer

Ce projet est fourni « en l'état », sans garantie. La responsabilité de l'usage des modèles
produits incombe entièrement à l'utilisateur.

- N'utilisez ce logiciel que sur des modèles dont la licence l'autorise.
- Toute diffusion d'un modèle modifié doit s'accompagner d'une model card (modèle de base, méthode,
  métriques).
- Le logiciel n'inclut aucune fonctionnalité destinée à produire du contenu gravement dangereux.

## Auteurs

Développé par QuelleEpoch et [LevelUp](https://www.levelup.run/).

## Licence

Distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour le texte complet.

Copyright © 2026 QuelleEpoch et LevelUp.
