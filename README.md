# Abliteration préservante

Outil d'abliteration (directional ablation) pour modèles de langage HuggingFace. Il retire la
direction de refus d'un modèle sans réentraînement, en cherchant à préserver deux capacités
qu'une abliteration naïve dégrade : la négation logique (« non, ce code est faux ») et les
capacités agentiques (appels d'outils, sorties structurées, raisonnement multi-étapes).

La méthode reprend la *directional ablation* d'Arditi et al. (2024) et sa variante *projected*
(orthogonalisation de la direction de refus contre des directions à préserver). Chaque étape du
pipeline est isolée, lisible et couverte par des tests.

## Principe

On calcule la direction de refus à partir d'un contraste de moyennes d'activations entre
prompts nuisibles et prompts neutres :

```
r = normalize(μ_harmful − μ_harmless)
```

Plutôt que d'effacer `r` directement, on l'orthogonalise d'abord contre les directions des
capacités à préserver (négation `n`, agentique `a`), de sorte que l'ablation ne touche que la
composante du refus indépendante de ces capacités :

```
r_safe = project_out(r, [n, a, ...])
W'     = W − r_safe (r_safeᵀ W)
```

L'orthogonalisation est appliquée à toutes les matrices qui écrivent dans le residual stream
(`o_proj`, `down_proj` de chaque couche, embeddings ; un `down_proj` par expert pour les MoE).

### Les quatre classes de prompts

| Classe | Rôle |
|---|---|
| `harmful` | Déclenche le refus ; combinée à `harmless`, donne la direction de refus |
| `harmless` | Référence neutre |
| `legitimate_negation` | Négation logique légitime, à préserver |
| `agentic` | Appels d'outils et sorties structurées, à préserver |

### Variantes d'ablation (`--variant`)

| Variante | Description |
|---|---|
| `conventional` | `W' = W − r(rᵀW)` ; efface la direction brute (baseline) |
| `projected` | Orthogonalise `r` contre `harmless` avant ablation |
| `preserving` | Orthogonalise `r` contre un sous-ensemble choisi de directions (défaut de production) |
| `norm_preserving_biprojected` | Idem `preserving`, avec préservation de la norme des poids |

La variante retenue par défaut est `preserving`, avec préservation de la négation et de
l'agentique.

## Architecture

Le pipeline ne dépend d'aucun nom de module codé en dur. Il passe par `ArchAdapter`, qui
localise les matrices écrivant dans le residual stream par introspection (`named_modules()`,
matching par suffixe). Sont gérées les architectures denses, les MoE (un `down_proj` par expert
plus les couches partagées) et les couches Conv1D de type GPT-2 (axes transposés).

```
abliteration/         Package Python
├── cli.py            Point d'entrée : python -m abliteration.cli <commande>
├── data/             Classes contrastives, chat template, holdout déterministe
├── models/           Chargement bf16 et ArchAdapter
├── directions/       Collecte d'activations, calcul des directions, sélection de couche
├── ablation/         project_out, variantes, orthogonalisation des poids, hooks réversibles
├── concepts/         Abstraction « concept » générique (direction, séparabilité, sonde)
├── eval/             Refus, KL, négation, agentique, benchmarks, juge LLM hors-ligne
├── circuits/         Analyse circuitielle (DLA + patching causal), lecture seule
├── optimize/         Objectif composite et boucle Optuna
├── io/               Sauvegarde safetensors et model card
├── cache.py          Cache disque des calculs déterministes
├── output.py         Contrat de sortie --json
└── heal.py           Récupération agentique par LoRA SFT

scripts/              Scripts d'expérimentation et de reproduction
tests/                Suite pytest (miroir du package)
data/                 Les quatre fichiers de prompts (JSONL)
results/              Rapports de mesure (scores agrégés)
```

Flux du pipeline :

```
modèle + 4 classes (chat template appliqué)
  → extract : moyennes d'activations par couche, dernier token → directions
  → select  : couche d'ablation retenue par séparabilité
  → apply   : orthogonalisation de toutes les écritures résiduelles (bf16)
  → eval    : rapport refus (holdout) + préservation (KL, négation, agentique)
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

La quantification 4-bit n'est utilisée que pour mesurer de gros modèles. Les poids ablitérés
sont toujours produits en bf16.

## Format des données

Un fichier JSONL par classe. Chaque ligne est un objet avec une clé `text` (ou `prompt`) ;
toute autre clé est conservée dans `meta` (utile pour `agentic`, qui peut porter le schéma
d'outil attendu). Le dossier passé via `--data-dir` doit contenir un fichier par classe :

```
data/
├── harmful.txt
├── harmless.txt
├── legitimate_negation.txt
└── agentic.txt
```

L'extension est `.txt` mais le contenu est du JSONL. Exemple pour `agentic` :

```json
{"text": "Appelle l'outil météo pour Paris", "tool": {"name": "get_weather", "parameters": {"required": ["city"]}}}
```

Chaque classe est découpée en train et holdout de façon déterministe : le train sert à calculer
les directions, le holdout à mesurer le refus sur des prompts non vus.

## Utilisation

Toutes les commandes prennent le modèle en argument positionnel (identifiant HuggingFace ou
chemin local). Options communes : `--data-dir`, `--dtype` (défaut `bfloat16`), `--device`,
`--batch-size`, `--holdout`, `--seed`. L'option `--no-cache` désactive le cache disque.

Pipeline complet :

```bash
python -m abliteration.cli abliterate meta-llama/Llama-3.1-8B-Instruct \
    --variant preserving --preserve negation,agentic --data-dir data --out ./out
```

La commande enchaîne `extract` et `apply`, puis sauvegarde le modèle en safetensors et une
model card (modèle de base, méthode, directions préservées, métriques).

Étapes séparées :

```bash
python -m abliteration.cli extract <model> --data-dir data --out directions.pt
python -m abliteration.cli select  <model> --directions directions.pt
python -m abliteration.cli apply   <model> --directions directions.pt \
    --variant preserving --preserve negation,agentic --layer 14 --out ./out
```

Diagnostic (lecture seule) de la séparabilité des directions par couche :

```bash
python -m abliteration.cli diagnose <model> --directions directions.pt --layers 8-20
```

### Optimisation

```bash
python -m abliteration.cli optimize <model> --trials 50 \
    --lambda-kl 1.0 --lambda-neg 2.0 --lambda-syco 0.5 --lambda-agent 3.0
```

L'objectif Optuna co-minimise le refus et les dégradations de préservation :

```
objectif = refusal_rate
         + λ_kl   · KL(harmless)
         + λ_neg  · (1 − negation_retention)
         + λ_syco · follow_rate
         + λ_agent· (1 − agentic_score)
```

Le terme `λ_agent` évite qu'un optimum sur (refus, KL) masque un effondrement des appels
d'outils. La boucle persiste ses essais en JSONL et reprend après interruption.

### Évaluation

```bash
python -m abliteration.cli eval ./out --benchmarks mmlu,gsm8k --out report.json
```

Le rapport couvre deux axes :

- Refus : `refusal_rate` sur le holdout, avec filtre de dégénérescence.
- Préservation : `kl`, `negation_retention`, `agentic_score`.
- Garde-fous : `degeneracy_rate`, `empty_rate`, `follow_rate`.

Le juge de refus par défaut est heuristique (mots-clés, déterministe, auditable). Il manque les
refus déguisés (un préambule « Sure, here's how... » suivi de rien, une déflexion moralisatrice).
Un juge LLM hors-ligne (`abliteration/eval/llm_judge.py`) permet de reclasser après coup des
sorties déjà générées en `REFUSAL`, `NON_REFUSAL` ou `EVASIVE`. Ce juge n'intervient pas dans le
pipeline de production et n'est fiable qu'après validation contre des labels humains (voir plus
bas).

### Récupération agentique

```bash
python -m abliteration.cli heal ./out --traces traces.jsonl --n-traces 200
```

À utiliser si l'évaluation révèle un effondrement agentique résiduel : un court LoRA SFT sur une
centaine à quelques centaines de traces d'appels d'outils restaure l'agentique sans réintroduire
le refus. Nécessite l'extra `heal` (peft).

### Analyse de concepts

L'abstraction `concepts/` généralise la direction de refus à tout concept défini par un
contraste de prompts. Trois concepts sont prédéfinis (`refusal`, `negation`, `agentic`) ; on peut
aussi en charger un depuis deux fichiers. Toutes ces commandes sont en lecture seule.

```bash
python -m abliteration.cli concept-direction    <model> --concept refusal
python -m abliteration.cli concept-separability  <model> --concepts refusal,negation,agentic
python -m abliteration.cli concept-probe         <model> --concept refusal
python -m abliteration.cli concept-steer         <model> --concept refusal --alpha 8
```

### Pilotage par programme

Toutes les commandes acceptent `--json`, qui renvoie sur stdout une enveloppe stable
`{schema_version, status, command, data, error}` (logs sur stderr, codes de sortie 0/1/2). La
commande `schema --json` décrit la CLI complète. Voir `AGENTS.md`.

## État d'implémentation

| Composant | État |
|---|---|
| `data`, `models`, `directions`, `ablation` | Implémenté et testé |
| `eval` (refus, KL, négation, agentique, juge hors-ligne) | Implémenté et testé |
| `concepts`, `cache`, contrat `--json` | Implémenté et testé |
| `optimize` (objectif composite, boucle Optuna) | Implémenté et testé |
| `heal` (LoRA SFT) | Implémenté ; nécessite l'extra `heal` |
| `io` (safetensors, model card) | Implémenté et testé |
| `circuits` (Phase 1, lecture seule) | Implémenté et testé |
| CLI (14 sous-commandes) | Câblée aux modules réels |
| `io.export_gguf` | Non implémenté (nécessite llama.cpp) |
| Analyse circuitielle Phase 2 (ablation ciblée) | Non implémenté |

### Résultats mesurés

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
environ 90 % de compliances réelles et deux refus déguisés que les mots-clés ne détectent pas,
soit un taux de refus effectif autour de 7 %.

## Tests

```bash
pytest                  # suite complète (~230 tests)
pytest -m "not model"   # exclut les tests qui chargent un modèle
ruff check .            # lint
```

Le test d'intégration `tests/integration/test_constraints.py` utilise un modèle jouet dont les
matrices d'écriture valent l'identité, ce qui rend la rétention d'une sonde calculable exactement
(`rétention = 1 − (d·p)²`). Il vérifie, sur la chaîne réelle
`compute_directions → ablation_direction → orthogonalize_weights → ArchAdapter`, que `preserving`
conserve la négation et l'agentique là où `conventional` les dégrade, le refus restant réduit
dans les deux cas.

### Validation du juge

Le `refusal_rate` ne vaut que ce que vaut le juge. Pour lever le doute sur le 0.00 heuristique,
les sorties ont été rejugées hors-ligne, puis le juge a été comparé à des labels humains :

- Le juge LLM 3B local n'a pas passé la validation. Avec une rubrique stricte, il classe en
  refus des réponses qui complient (biais de nocivité). Avec une rubrique few-shot débiaisée, il
  devient bon sur le modèle ablitéré mais aveugle aux refus francs du modèle de base. La référence
  reste donc le label humain.
- La lecture humaine confirme que l'ablation est solide : le 0.00 heuristique était optimiste,
  pas trompeur.

Deux règles en découlent : valider tout juge automatique sur un échantillon humain, et conserver
les textes bruts des générations (pas seulement les scores) pour pouvoir rejuger. Ces textes
peuvent contenir du contenu nuisible et restent donc hors du dépôt.

## Points d'attention

Sources de bugs silencieux en abliteration :

- Appliquer le chat template avant toute collecte d'activations, sinon la direction est bruitée.
- Padding à gauche, ou indexation par `attention_mask`, pour capter le dernier token.
- Orthogonaliser toutes les écritures résiduelles (`o_proj`, `down_proj`, embeddings ; chaque
  expert pour les MoE) ; en oublier laisse du refus résiduel.
- Produire les poids en bf16 ; la 4-bit est réservée à la mesure.
- Passer par `ArchAdapter` plutôt que de coder des noms de modules en dur.
- Utiliser les hooks réversibles (`abliteration/ablation/hooks.py`) pour explorer et sélectionner
  une couche ; l'orthogonalisation permanente n'intervient qu'à la livraison.

## Cadre d'usage

L'abliteration est une technique d'interprétabilité publiée et à double usage. Ce dépôt reste un
outil générique de modification de modèle :

- Toute livraison de poids s'accompagne d'une model card (modèle de base, méthode, métriques).
- L'évaluation sur les deux axes fait partie intégrante du pipeline.
- Aucune fonctionnalité n'est orientée vers la production de contenu gravement dangereux.

## Analyse circuitielle (Phase 1)

Au-delà de la direction de refus, `abliteration/circuits/` cherche quels composants (têtes
d'attention, MLP) portent le refus et comment l'information circule. La Phase 1 est en lecture
seule : aucune modification de poids. La règle suivie est que la DLA, corrélationnelle, ne conclut
jamais seule ; toute localisation est confirmée par patching causal (nécessité et suffisance).

| Module | Rôle |
|---|---|
| `backend.py` | Introspection par composant : `TorchHookBackend` (read/write) et `NNsightBackend` (read) |
| `dla.py` | Direct Logit Attribution (corrélationnel) |
| `patching.py` | Activation patching causal au dernier token (knockout et restauration) |
| `attribution.py` | Attribution par gradient, vérifiée sur les top-k par patching exact |
| `localize.py` | Agrégation DLA + patching, stabilité bootstrap (Jaccard), métriques de fidélité |
| `report.py` | Rapport séparant corrélationnel et causalement validé |

```bash
python -m abliteration.cli analyze-circuit Qwen/Qwen3-0.6B --device cuda \
    --pairs 16 --top-k 24 --threshold 0.5 --n-boot 300 --out rapport.json
```

Le backend par défaut est `TorchHookBackend` ; NNsight est un backend de lecture alternatif,
vérifié par un test de parité sur Qwen3-0.6B. La Phase 2 (ablation ciblée) n'est pas implémentée :
sur Qwen3-0.6B, la localisation s'est révélée instable d'un run à l'autre (Jaccard bootstrap
inférieur à 0.9), ce qui ne justifie pas d'y conditionner une ablation chirurgicale.

## Référence

Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.

Les hyperparamètres employés sont commentés à leur point d'usage. En cas de doute, se fier aux
tests et aux métriques mesurées.

## Disclaimer

Ce projet est un outil de recherche en interprétabilité, fourni à des fins éducatives et
défensives. L'abliteration est une technique à double usage : retirer la direction de refus d'un
modèle lève des garde-fous mis en place par ses auteurs. La responsabilité de l'usage des modèles
produits incombe entièrement à l'utilisateur.

- N'utilisez ce logiciel que sur des modèles dont la licence et les conditions d'utilisation
  l'autorisent.
- Toute diffusion d'un modèle modifié doit être accompagnée d'une model card indiquant le modèle
  de base, la méthode appliquée et les métriques d'évaluation.
- Le logiciel n'inclut aucune fonctionnalité destinée à produire du contenu gravement dangereux,
  et n'a pas vocation à en faciliter la production.

Le logiciel est fourni « en l'état », sans garantie d'aucune sorte. Les auteurs ne sauraient être
tenus responsables des dommages résultant de son utilisation.

## Licence

Distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour le texte complet.
