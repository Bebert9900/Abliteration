# Paysage des outils d'abliteration (2024–2026)

Carte des projets pour décider quoi réutiliser/forker plutôt que réimplémenter.

## Outils automatiques (optimisation de paramètres)

### Heretic — `p-e-w/heretic`
- Install : `pip install -U heretic-llm` puis `heretic <hf-model-id>`.
- Cœur : directional ablation + optimiseur **TPE (Optuna)** entièrement automatique.
- Objectif optimisé : co-minimiser (nb de refus sur prompts harmful) et (divergence **KL**
  vs modèle original sur prompts harmless) → modèle dé-censuré qui retient un max
  d'intelligence.
- Paramètres optimisés : `direction_index` (ou `per layer`), kernel de poids d'ablation
  (`max_weight`, `max_weight_position`, `min_weight`, `min_weight_distance`), séparément par
  composant.
- Supporte : dense, beaucoup de multimodaux, plusieurs MoE, hybrides (Qwen3). MXFP4 (gpt-oss)
  nécessite PyTorch ≥ 2.6 (`torch.accelerator`).
- Prérequis : Python 3.10+, PyTorch 2.2+. Gestion des deps via `uv` (uv.lock fourni).
- Recherche : `heretic --plot-residuals <model>` (plots PaCMAP des résidus),
  `heretic --print-residual-geometry <model>`.
- Licence : AGPL-3.0.

### Abliterix — `wuwangzhang1216/abliterix`
- Dérivé de Heretic (AGPL-3.0). Ajoute :
  - Ablation **expert-granular pour MoE** (Mixtral, Qwen3 MoE, DeepSeek, Phi-3.5-MoE,
    Granite MoE, DBRX, Llama-4, gpt-oss MXFP4).
  - Support dense, MoE, SSM/hybride, vision-language ; 150+ configs préfaites (Llama, Gemma,
    Phi, Qwen, Mistral, Yi, InternLM, Falcon, Cohere, EXAONE, Granite, OLMo, SmolLM, SOLAR…).
  - LoRA, steering direct.
  - **HonestAbliterationBench** : benchmark reproductible avec contrat figé
    (`min_new_tokens=100`, `max_new_tokens=150`, greedy, juge LLM avec filtre de dégénérescence,
    KL vs base déclarée). Résiste aux deux modes d'échec classiques des leaderboards
    (générations trop courtes + juges par mots-clés).

### Blasphemer — `sunkencity999/blasphemer`
- Fork de Heretic optimisé macOS/Apple Silicon, +checkpoint/resume, LoRA fine-tuning
  (injection de connaissances), conversion GGUF en une commande. ~55% plus rapide sur Mac.

## Outils manuels (contrôle fin)

### llm-abliteration — `jim-plus/llm-abliteration` (miroir `NousResearch/llm-abliteration`)
- Basé sur `transformers` (descend du code de Sumandora ; accède à `post` via `hidden_states`).
- Flags clés : `--projected` (orthogonaliser la direction de refus à la mesure),
  `--normpreserve` (préserver les normes à l'ablation), `--invert` (passer d'ablation à
  addition → re-censurer / tester).
- Datasets de prompts custom : `.txt`, `.parquet`, `.json`, `.jsonl`, locaux ou depuis HF.
- 4-bit (bitsandbytes) possible pour mesurer sur gros modèles, mais l'ablation finale doit se
  faire en pleins poids (bf16). Config YAML pour multi-couches sources et stratégies par
  couche destination. `-c` produit des charts pour repérer les couches candidates.

### OBLITERATUS — `elder-plinius/OBLITERATUS`
- Cross-hardware / cross-model / cross-method ; gère Conv1D (GPT-2) et Linear, attention
  standard et fusionnée, archis custom via `trust_remote_code`. Builder de config visuel
  (`docs/index.html`). Vise un grand dataset de télémétrie d'abliteration.

### ErisForge / DECCP
- ErisForge : wrappers runtime. DECCP : single-pass quantifié, bon profil de préservation
  (orthogonalisation norm-preserving). Bons pour dégradation minimale des capacités.

## Références académiques

- Arditi et al. 2024 — *Refusal in LMs Is Mediated by a Single Direction* (arXiv:2406.11717).
- Zhao et al. 2025 — *LLMs Encode Harmfulness and Refusal Separately*.
- grimjim 2025 — *Projected Abliteration* ; *Norm-Preserving Biprojected Abliteration*
  (HuggingFace blog).
- Agnihotri et al. 3 oct. 2025 — *A Granular Study of Safety Pretraining under Model
  Abliteration* (arXiv:2510.02768) ; code `shashankskagnihotri/safety_pretraining`.
- Gülmez 2026 — *Gabliteration* (arXiv:2512.18901).
- Abu Shairah et al. (KAUST) mai 2025 — *An Embarrassingly Simple Defense Against LLM
  Abliteration Attacks* (arXiv:2505.19056) — côté défense (extended-refusal fine-tuning).

## Tutoriel de référence

mlabonne, *Uncensor any LLM with abliteration* (HuggingFace blog) — l'explication pédagogique
canonique + notebook Colab. Bon point d'entrée pour comprendre la chaîne complète.
