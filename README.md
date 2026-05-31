# Abliteration préservante

> Outil d'**abliteration** de LLM (directional ablation) qui retire le **refus moral**
> d'un modèle HuggingFace **sans réentraînement** — tout en **préservant** deux capacités
> qu'une abliteration naïve détruit : la **négation logique légitime** (« non, ce code est
> faux ») et les **capacités agentiques** (tool use, appels de fonctions, multi-étapes).

Projet **from scratch, pédagogique et réutilisable** : chaque étape est lisible, testée
(68 tests, TDD) et conforme aux sources de vérité du repo (`CLAUDE.md`,
`ABLITERATION_KNOWLEDGE_BASE.md`, les 3 skills `.claude/skills/`, et la spec
`docs/superpowers/specs/2026-05-30-abliteration-tool-design.md`).

---

## 1. Le problème, en deux phrases

L'abliteration classique (Arditi et al. 2024) identifie une **« direction de refus »** dans
le residual stream et l'efface des poids. Supprimer le refus est facile ; le faire **sans
lobotomiser le modèle** est tout l'enjeu — et la version naïve abîme silencieusement des
capacités voisines de la direction de refus, notamment la négation légitime et l'agentique.

## 2. L'idée centrale : l'abliteration *préservante*

On part de la **direction de refus** canonique, calculée sur un contraste de moyennes
d'activations :

```
r̂ = normalize( μ_harmful − μ_harmless )
```

Au lieu d'effacer `r̂` telle quelle, on l'**orthogonalise d'abord contre les directions à
préserver** (généralisation de la *projected abliteration*). On retire de `r̂` toute
composante qui pointe vers la négation `n̂` ou l'agentique `â` :

```
r̂_safe = project_out( r̂, contre [n̂, â, …] )
```

Ainsi l'ablation ne touche plus que la part du refus **orthogonale** aux capacités qu'on
veut garder. C'est la thèse du projet, et elle est **prouvée par un test d'intégration de
bout en bout** (voir §8).

### Les 4 classes contrastives

| Classe | Rôle |
|---|---|
| `harmful` | Déclenche le refus (avec `harmless`, donne `r̂`) |
| `harmless` | Référence neutre / baseline |
| `legitimate_negation` | Négation logique légitime — **à préserver** (`n̂`) |
| `agentic` | Tool use, schéma strict, multi-étapes — **à préserver** (`â`) |

### Les variantes d'ablation (`--variant`)

| Variante | Ce qu'elle fait |
|---|---|
| `conventional` | `W' = W − r̂(r̂ᵀW)` — baseline, efface `r̂` telle quelle |
| `projected` | orthogonalise `r̂` contre `harmless` avant ablation |
| `preserving` | orthogonalise `r̂` contre un sous-ensemble choisi `[n̂, â, …]` |
| `norm_preserving_biprojected` | préserve aussi la norme des poids (raffinement, KB §3.4) |

## 3. Architecture

Tout le pipeline est **agnostique à l'architecture** : il ne parle qu'à `ArchAdapter`, qui
localise les matrices écrivant dans le residual stream **sans coder les noms de modules en
dur** (balayage `named_modules()`, matching par suffixe). Gère dense, MoE (un `down_proj`
par expert + experts partagés) et Conv1D (GPT-2, axes transposés).

```
src/
├── cli.py        # point d'entrée : python -m src.cli <sous-commande>
├── data/         # 4 classes contrastives, chat template, padding gauche, holdout
├── models/       # chargement bf16 + ArchAdapter (introspection d'archi)
├── directions/   # collecte d'activations, directions 4 classes, séparabilité, sélection
├── ablation/     # project_out, variantes, orthogonalisation des poids, hooks réversibles
├── eval/         # refus, KL, négation, agentique, benchmarks, rapport bi-axe
├── optimize/     # objectif composite + boucle Optuna TPE
├── io/           # safetensors + model card transparente (+ export GGUF stub)
├── heal.py       # réparation agentique post-abliteration (stub documenté)
└── circuits/     # analyse circuitielle + ablation chirurgicale ciblée — PLANIFIÉ (non implémenté)
```

### Pistes de conception documentées (skills)

Au-delà du code, le repo embarque des **skills** (`.claude/skills/`) qui figent le vocabulaire,
les maths et les choix d'algorithme. L'un d'eux, **`abliteration-circuits`**, décrit une
direction de travail non encore codée : l'**analyse circuitielle** (Direct Logit Attribution,
activation/attribution patching) pour localiser *quels composants* portent le refus, puis une
**ablation chirurgicale ciblée** (`circuit_targeted`) n'intervenant que sur les ~3 % de têtes/MLP
causalement responsables — hypothèse à tester : préserve-t-elle mieux les capacités que
l'orthogonalisation large ? Règle d'or du skill : analyse validée causalement **avant** toute
ablation ciblée.

### Flux de données

```
modèle + 4 classes (chat template appliqué)
   └─(extract)→ moyennes d'activations par couche, dernier token → directions r̂/n̂/â/ĥ
        └─(select)→ couche d'ablation retenue (séparabilité r̂ vs n̂/â)
             └─(apply, bf16)→ orthogonalisation de TOUTES les écritures résiduelles
                  └─(eval)→ rapport bi-axe : refus (holdout) + préservation (KL/négation/agentique)
```

## 4. Installation

Stack : **Python ≥ 3.10**, **PyTorch ≥ 2.2** (CUDA pour tout sauf les très petits modèles).
La gestion de dépendances cible est **`uv`** ; à défaut, `pip` fonctionne aussi.

```bash
# Avec uv (recommandé)
uv sync

# Ou avec pip
pip install -e .

# Groupes optionnels (installés à la demande)
pip install -e ".[optimize]"   # optuna
pip install -e ".[quant]"      # bitsandbytes — MESURE uniquement, jamais pour livrer
pip install -e ".[eval]"       # lm-eval — MMLU/GSM8K/…
pip install -e ".[dev]"        # pytest
```

> ⚠️ **bf16 pour livrer.** La quantification 4-bit est tolérée pour *mesurer* sur de gros
> modèles, **jamais** pour figer les poids abliteré. L'ablation finale est en bf16.

## 5. Format des données

Un fichier **JSONL** par classe, une ligne = un objet JSON avec une clé `text` (ou
`prompt`). Toute autre clé est conservée dans `meta` (utile pour `agentic` : schéma d'outil
attendu, appel de référence). Le dossier passé via `--data-dir` doit contenir un fichier
nommé d'après la valeur de chaque classe :

```
data/
├── harmful.txt
├── harmless.txt
├── legitimate_negation.txt
└── agentic.txt
```

> Note : l'extension est `.txt` mais le **contenu est du JSONL**. Exemple de ligne pour
> `agentic` :
> ```json
> {"text": "Appelle l'outil météo pour Paris", "tool": {"name": "get_weather", "parameters": {"required": ["city"]}}}
> ```

Chaque classe est découpée en **train / holdout** déterministe : le `train` sert à calculer
les directions, le `holdout` à mesurer le refus sur des prompts **jamais vus** (sinon on
sur-estime le succès).

## 6. Mode d'emploi (CLI)

Toutes les sous-commandes prennent le modèle en argument positionnel (identifiant HF ou
chemin local). Options communes : `--data-dir`, `--dtype` (défaut `bfloat16`), `--device`,
`--batch-size`, `--holdout`, `--seed`.

### Pipeline complet (consolidé)

```bash
python -m src.cli abliterate meta-llama/Llama-3.1-8B-Instruct \
    --variant preserving \
    --preserve negation,agentic \
    --data-dir data \
    --out ./out
```

Enchaîne `extract → apply`, sauvegarde le modèle en safetensors **et** une **model card
transparente** (modèle de base + méthode + directions préservées + métriques) — exigée par
le cadre responsable du projet.

### Étapes granulaires (transparence pédagogique)

```bash
# 1. Calcule et sauvegarde les directions des 4 classes
python -m src.cli extract <model> --data-dir data --out directions.pt

# 2. Sélectionne la meilleure couche d'ablation (séparabilité)
python -m src.cli select <model> --directions directions.pt

# 3. Applique l'ablation orthogonalisée et sauvegarde le modèle
python -m src.cli apply <model> --directions directions.pt \
    --variant preserving --preserve negation,agentic --layer 14 --out ./out
```

### Diagnostic (lecture seule)

```bash
# Inspecte la séparabilité r̂ vs n̂/â par couche — repère les couches à risque de débordement
python -m src.cli diagnose <model> --directions directions.pt --layers 8-20
```

### Optimisation des poids λ (Optuna)

```bash
python -m src.cli optimize <model> --trials 50 \
    --lambda-kl 1.0 --lambda-neg 2.0 --lambda-syco 0.5 --lambda-agent 3.0
```

L'objectif composite co-minimise refus **et** dégradations de préservation :

```
objectif = refusal_rate
         + λ_kl   · KL(harmless)
         + λ_neg  · (1 − negation_retention)
         + λ_syco · follow_rate            (sycophantie / capitulation indue)
         + λ_agent· (1 − agentic_score)
```

Sans `λ_agent`, l'optimiseur peut livrer un modèle qui hallucine ses tool calls tout en
affichant un excellent (refus, KL) — d'où les termes étendus.

### Évaluation

```bash
python -m src.cli eval ./out --benchmarks mmlu,gsm8k --out report.json
```

Produit un **rapport bi-axe** :
- **Axe refus** : `refusal_rate` sur le holdout (juge + filtre de dégénérescence).
- **Axe préservation** : `kl`, `negation_retention`, `agentic_score`.
- **Garde-fous anti-gaming** : `degeneracy_rate`, `empty_rate`, `follow_rate`.

### Réparation agentique (`heal`)

```bash
python -m src.cli heal ./out --traces traces.jsonl --n-traces 200
```

À lancer **uniquement** si l'éval révèle un effondrement agentique résiduel malgré
`preserving` : un court LoRA SFT sur ~100–300 traces de tool use restaure l'agentique sans
réintroduire le refus.

## 7. État d'implémentation (honnête)

Tout le **cœur algorithmique** est implémenté et testé. Deux handlers CLI et deux exports
restent des **stubs documentés** (interface posée, câblage réel à brancher) :

| Composant | État |
|---|---|
| `data`, `models`/`ArchAdapter`, `directions`, `ablation` | ✅ implémenté + testé |
| `eval` (métriques refus/KL/négation/agentique, rapport) | ✅ implémenté + testé |
| `optimize` (objectif composite, boucle Optuna) | ✅ implémenté + testé |
| `io` (safetensors, model card) | ✅ implémenté + testé |
| CLI `extract` / `select` / `apply` / `abliterate` / `diagnose` | ✅ câblés aux vrais modules |
| CLI `eval` / `optimize` (handlers) | ⚠️ **partiellement stubés** : construisent un rapport/objectif à zéros — le câblage génération-sur-holdout reste à finir |
| `heal()` | ⚠️ **stub** : lève `NotImplementedError` avec la marche à suivre |
| `io.export_gguf` | ⚠️ **stub** : nécessite llama.cpp |

> Conséquence pratique : `abliterate`/`apply`/`extract` produisent un vrai modèle abliteré ;
> `eval` et `optimize` ne renvoient pas encore de métriques réelles tant que la génération
> sur holdout n'est pas branchée.

## 8. Tests

```bash
pytest          # 68 tests, ~2 s
```

Le test phare est `tests/integration/test_constraints.py` : sur un **modèle jouet torch**
dont les writers résiduels valent l'identité, la rétention d'une sonde devient
**mesurable exactement** (`rétention = 1 − (d·p)²`). Il prouve, via la vraie chaîne
`compute_directions → ablation_direction → orthogonalize_weights → ArchAdapter`, que :

- `preserving` **garde** négation et agentique (rétention ≈ 1.0) ;
- `conventional` les **abîme** (rétention < 0.95) au même endroit ;
- dans les deux cas le **refus est bien réduit** (canari anti-régression).

## 9. Gotchas critiques (sources de bugs silencieux)

- **Chat template systématique** avant collecte d'activations (sinon `r̂` est bruitée).
- **Padding à gauche** / indexation par `attention_mask` pour le « dernier token ».
- **Orthogonaliser TOUTES les écritures résiduelles** : `o_proj` + `down_proj` (toutes
  couches) + embeddings ; pour MoE, chaque expert + les partagées. En oublier → refus résiduel.
- **bf16 pour livrer**, 4-bit pour mesurer seulement.
- **Ne jamais coder les noms de modules en dur** — passer par `ArchAdapter`.
- **Hooks réversibles** (`src/ablation/hooks.py`) pour explorer/sélectionner ;
  orthogonalisation permanente seulement pour livrer.

## 10. Cadre responsable

L'abliteration est une technique d'interprétabilité publiée et **dual-use**. Ce dépôt reste
un **outil générique de modification de modèle** :

- **Model card obligatoire** à toute livraison de poids (base + méthode + métriques).
- L'**évaluation** (les deux axes) et la **recherche défensive** font partie du projet, pas
  des options.
- Aucune fonctionnalité orientée vers la production de contenu gravement dangereux.

## 11. Références

- Arditi et al. 2024 — *Refusal in LLMs is mediated by a single direction* (technique de base,
  KB §2).
- `arXiv:2603.27518`, `arXiv:2604.08388`, et le chiffre « GSM8K −18,81 pp » sont **fournis par
  l'utilisateur et hors de la base de connaissances v.mai-2026** — à confirmer avant de s'en
  prévaloir.

Pour tout détail factuel (maths des variantes, hyperparamètres, état de l'art),
`ABLITERATION_KNOWLEDGE_BASE.md` est la source de vérité figée du projet.
