# Abliteration préservante

> Outil d'**abliteration** de LLM (directional ablation) qui retire le **refus moral**
> d'un modèle HuggingFace **sans réentraînement** — tout en **préservant** deux capacités
> qu'une abliteration naïve détruit : la **négation logique légitime** (« non, ce code est
> faux ») et les **capacités agentiques** (tool use, appels de fonctions, multi-étapes).

Projet **from scratch, pédagogique et réutilisable** : chaque étape est lisible et testée
(TDD). La méthode s'appuie sur la *directional ablation* (Arditi et al. 2024) et sa
généralisation *projected* (orthogonalisation de la direction de refus contre les directions
à préserver).

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

Disposition du dépôt (layout professionnel) :

```
abliteration/         # le package Python (importable, testé)
├── cli.py            # point d'entrée : python -m abliteration.cli <sous-commande>
├── data/             # 4 classes contrastives, chat template, padding gauche, holdout déterministe
├── models/           # chargement bf16 + ArchAdapter (introspection d'archi)
├── directions/       # collecte d'activations, directions 4 classes, séparabilité, sélection
├── ablation/         # project_out, variantes, orthogonalisation des poids, hooks réversibles
├── eval/             # refus (heuristique + juge LLM hors-ligne), KL, négation, agentique, benchmarks
├── circuits/         # analyse circuitielle (DLA + patching causal) — Phase 1, lecture seule
├── optimize/         # objectif composite + boucle Optuna TPE
├── io/               # safetensors + model card transparente (+ export GGUF stub)
└── heal.py           # réparation agentique post-abliteration (stub documenté)

scripts/              # scripts d'expérimentation/repro (run depuis la racine)
│                     #   chat.py · run_benchmarks.py · rejudge_harmful.py · compare_variants.py
tests/                # suite pytest (156 tests), miroir de l'arbo du package
data/                 # les 4 fichiers de prompts (.txt au format JSONL)
results/              # rapports de mesure suivis (JSON de scores) ; *_generations.json ignorés (harmful)
artifacts/            # modèles abliteré produits (gitignoré — jamais de poids dans git)
docs/                 # rapports d'expériences (privés, gitignorés)
pyproject.toml        # déps + entry point console `abliterate`
```

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
python -m abliteration.cli abliterate meta-llama/Llama-3.1-8B-Instruct \
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
python -m abliteration.cli extract <model> --data-dir data --out directions.pt

# 2. Sélectionne la meilleure couche d'ablation (séparabilité)
python -m abliteration.cli select <model> --directions directions.pt

# 3. Applique l'ablation orthogonalisée et sauvegarde le modèle
python -m abliteration.cli apply <model> --directions directions.pt \
    --variant preserving --preserve negation,agentic --layer 14 --out ./out
```

### Diagnostic (lecture seule)

```bash
# Inspecte la séparabilité r̂ vs n̂/â par couche — repère les couches à risque de débordement
python -m abliteration.cli diagnose <model> --directions directions.pt --layers 8-20
```

### Optimisation des poids λ (Optuna)

```bash
python -m abliteration.cli optimize <model> --trials 50 \
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
python -m abliteration.cli eval ./out --benchmarks mmlu,gsm8k --out report.json
```

Produit un **rapport bi-axe** :
- **Axe refus** : `refusal_rate` sur le holdout (juge + filtre de dégénérescence).
- **Axe préservation** : `kl`, `negation_retention`, `agentic_score`.
- **Garde-fous anti-gaming** : `degeneracy_rate`, `empty_rate`, `follow_rate`.

Le juge de refus par défaut est **heuristique** (mots-clés, déterministe, `KeywordRefusalJudge`) :
rapide et auditable, mais il **rate les refus déguisés** (un « Sure, here's how… » suivi de rien,
une déflexion moralisatrice). Pour lever ce doute, un **juge LLM hors-ligne** (`src/eval/llm_judge.py`)
permet de **re-classer après coup** les sorties déjà générées en `REFUSAL / NON_REFUSAL / EVASIVE`.

> ⚠️ Cadre : ce juge LLM est une **analyse hors-ligne** sur des sorties déjà produites, **pas** une
> dépendance du pipeline qui produit le modèle (lequel reste sans IA au runtime). Et un juge LLM
> n'est fiable que **validé** : voir §8.1 (un petit juge 3B local a échoué la validation humaine).

### Réparation agentique (`heal`)

```bash
python -m abliteration.cli heal ./out --traces traces.jsonl --n-traces 200
```

À lancer **uniquement** si l'éval révèle un effondrement agentique résiduel malgré
`preserving` : un court LoRA SFT sur ~100–300 traces de tool use restaure l'agentique sans
réintroduire le refus.

## 7. État d'implémentation (honnête)

Tout le **cœur algorithmique** est implémenté et testé, et la CLI produit désormais un **vrai
modèle abliteré et de vraies métriques** (génération sur holdout branchée). Restent des stubs
documentés sur deux exports/réparation.

| Composant | État |
|---|---|
| `data`, `models`/`ArchAdapter`, `directions`, `ablation` | ✅ implémenté + testé |
| `eval` (refus/KL/négation/agentique, juge LLM hors-ligne, rapport) | ✅ implémenté + testé |
| `optimize` (objectif composite, boucle Optuna) | ✅ implémenté + testé |
| `io` (safetensors, model card) | ✅ implémenté + testé |
| CLI `extract` / `select` / `apply` / `abliterate` / `diagnose` | ✅ câblés aux vrais modules |
| CLI `eval` | ✅ **réel** : génération sur holdout + métriques bi-axe (option `--base`, `--benchmarks`) |
| `analyze-circuit` | ✅ réel (lecture seule, voir §12) |
| `cmd_optimize` (handler) | ⚠️ objectif encore stubé (renvoie `0.0`) — la boucle Optuna existe, le branchement génération reste à finir |
| `heal()` | ⚠️ **stub** : lève `NotImplementedError` avec la marche à suivre |
| `io.export_gguf` | ⚠️ **stub** : nécessite llama.cpp |

### 7.1 Validation empirique (Qwen2.5-3B-Instruct, variante `preserving`, couche 22)

Mesuré sur le holdout harmful (30 prompts), juge **heuristique** :

| métrique | base | abliteré |
|---|---|---|
| `refusal_rate` | 0.90 | **0.00** |
| `negation_retention` | 0.93 | 0.90 |
| `agentic_score` | 0.97 | 1.00 |
| `degeneracy` / `empty` | 0 / 0 | 0 / 0 |
| KL de préservation | — | 0.81 |

Refus supprimé, capacités préservées, pas de lobotomie.

## 8. Tests

```bash
pytest                    # 156 tests, ~6 s
pytest -m "not model"     # idem en sautant les tests qui chargent un vrai modèle
```

Le test phare est `tests/integration/test_constraints.py` : sur un **modèle jouet torch**
dont les writers résiduels valent l'identité, la rétention d'une sonde devient
**mesurable exactement** (`rétention = 1 − (d·p)²`). Il prouve, via la vraie chaîne
`compute_directions → ablation_direction → orthogonalize_weights → ArchAdapter`, que :

- `preserving` **garde** négation et agentique (rétention ≈ 1.0) ;
- `conventional` les **abîme** (rétention < 0.95) au même endroit ;
- dans les deux cas le **refus est bien réduit** (canari anti-régression).

### 8.1 Valider le juge avant de lui faire confiance

Le `refusal_rate` ne vaut que ce que vaut le **juge**. Le juge heuristique (mots-clés) est le
maillon faible : un `0.00` peut signifier « ne refuse vraiment plus » **ou** « le juge ne voit pas
les refus déguisés ». On a levé le doute en re-jugeant les sorties déjà générées avec un juge LLM
hors-ligne (`rejudge_harmful.py`), **puis en validant ce juge contre des labels humains** :

- Le **juge LLM 3B local a échoué** la validation. Rubrique stricte → biais de nocivité (classe
  REFUSAL des réponses qui *complient* ; accord ↔ humain = 6.7 %). Rubrique few-shot dé-biaisée →
  bonne sur l'abliteré (86.7 %) mais **aveugle aux refus francs** côté base (46.7 %). Conclusion :
  remplacer un instrument douteux par un juge 3B instable serait une erreur — **la référence
  autoritaire reste le label humain**, et un juge LLM fiable demande un modèle plus grand/dédié.
- **Verdict sur le 0 %** : en lisant les 30 réponses abliterées, **90 % sont de vraies
  compliances**. Le 0 % heuristique ne masque pas un mur de refus déguisés ; il était seulement
  un peu optimiste — le taux de refus humain autoritaire est **≈ 6.7 %** (2 refus déguisés que
  les mots-clés ratent) + 3.3 % évasif. L'abliteration est confirmée solide.

> Leçon réutilisable : **toujours valider un juge automatique sur un échantillon humain** par run,
> et sauvegarder les **textes bruts** des générations (pas seulement les scores) pour pouvoir
> re-juger après coup. Les générations brutes peuvent contenir du contenu harmful → `gitignore`.

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
  l'utilisateur et hors d'une base de connaissances vérifiée — à confirmer avant de s'en
  prévaloir.

Les chiffres et hyperparamètres employés dans le code sont commentés à leur point d'usage ;
en cas de doute, se fier aux tests et aux métriques mesurées plutôt qu'à des valeurs annoncées.

## 12. Analyse circuitielle du refus (`src/circuits/`, Phase 1)

Au-delà de l'analyse *directionnelle* (« quelle direction »), `src/circuits/` répond à
**« quels composants** (têtes d'attention, MLP) portent le refus, et comment l'information
circule ». **Phase 1 = analyse seulement, AUCUNE modification de poids.**

**Règle d'or** : la DLA (corrélationnelle) ne conclut jamais seule ; toute localisation est
**confirmée par patching causal** (nécessité + suffisance) avant d'être dite « validée ».

| Module | Rôle |
|---|---|
| `backend.py` | introspection par composant sur les **poids HF exacts** : `TorchHookBackend` (hooks torch, read+write) et `NNsightBackend` (trace nnsight, read-path) — décomposition exacte par tête |
| `dla.py` | Direct Logit Attribution (corrélationnel, marqué comme tel) |
| `patching.py` | activation patching causal **ciblé au dernier token** : nécessité (knockout) + suffisance (restauration) |
| `attribution.py` | attribution gradient scalable + contre-vérification des top-k par patching exact |
| `localize.py` | agrège DLA+patching → circuit *core* causal, stabilité bootstrap (Jaccard), faithfulness/CPR/CMD |
| `report.py` | rapport JSON/texte séparant **corrélationnel** vs **causalement validé** |

```bash
# Analyse circuitielle (lecture seule, produit un rapport) :
python -m abliteration.cli analyze-circuit Qwen/Qwen3-0.6B --device cuda \
    --pairs 16 --top-k 24 --threshold 0.5 --n-boot 300 --out rapport.json

# Backend nnsight (parité DLA) — nécessite l'extra circuits :
pip install -e ".[circuits]"
```

Backend par défaut : `TorchHookBackend` (couvre tout le pipeline sur les poids HF exacts) ;
NNsight est un backend de **lecture** alternatif, vérifié par un **test de parité**
DLA(torch) ≈ DLA(nnsight) sur Qwen3-0.6B (écart ~1e-5 par tête, bruit float32).

> **Phase 2 (ablation chirurgicale ciblée) NON implémentée** — conditionnée à une Phase 1
> *stable*. Sur Qwen3-0.6B, la localisation s'est révélée **instable** (le circuit core change
> d'un run à l'autre, bootstrap Jaccard < 0.9) : pas assez robuste pour décider la Phase 2 sur
> ce modèle. Résultat rapporté honnêtement — le champ est jeune et faillible.
