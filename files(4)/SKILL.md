---
name: abliteration-core
description: Connaissances de référence sur l'abliteration (directional ablation) des LLM — la théorie, les maths du calcul de la direction de refus, et le catalogue des variantes (conventionnelle, projected, norm-preserving biprojected, gabliteration) et des outils (Heretic, Abliterix, llm-abliteration). Utilise ce skill dès que le projet touche à l'abliteration, à la suppression de refus/safety alignment, aux "refusal directions", à l'orthogonalisation de poids, ou au choix d'une variante/d'un outil. À consulter avant d'écrire du code de pipeline pour fixer le vocabulaire et les choix d'algorithme.
---

# Abliteration — noyau théorique et catalogue

Ce skill fixe le vocabulaire, les maths et les choix d'algorithme. Pour l'implémentation
concrète, voir le skill `abliteration-pipeline`. Pour mesurer la qualité, voir
`abliteration-eval`.

## Le principe en une phrase

Arditi et al. (2024, *Refusal in Language Models Is Mediated by a Single Direction*,
arXiv:2406.11717) ont montré que le comportement de refus d'un LLM aligné est porté par
**une direction unique dans le residual stream**. Supprimer la capacité du modèle à
représenter cette direction supprime sa capacité à refuser ; l'ajouter artificiellement
provoque des refus même sur des requêtes anodines.

L'abliteration = trouver cette direction puis l'« ablater » des poids ou des activations.
C'est une technique d'édition de modèle / interprétabilité mécanistique, **sans
réentraînement**.

## L'algorithme conventionnel (référence)

Cinq étapes. Les détails d'implémentation sont dans `abliteration-pipeline`.

1. **Deux datasets contrastifs** : un ensemble de prompts *harmful* (qui déclenchent le
   refus) et un ensemble *harmless* (qui ne le déclenchent pas), idéalement équilibrés et
   appariés en style/longueur.
2. **Collecte d'activations** : passer les deux ensembles dans le modèle et mettre en cache
   les activations du residual stream, par couche, à la position du dernier token de
   l'instruction (ou moyennées sur les positions de l'instruction).
3. **Direction de refus** : pour chaque couche `l`,
   `r_l = mean(activations_harmful_l) − mean(activations_harmless_l)`, puis normaliser en
   vecteur unitaire `r̂_l`.
4. **Sélection de la couche/direction** : tester les directions candidates ; les meilleures
   se situent en général dans les **couches du milieu à milieu-tardives**. On retient la
   direction qui réduit le plus le refus quand on l'ablate, sans casser le modèle.
5. **Ablation**, deux modalités :
   - **Inference-time (hooks)** : retrancher la projection sur `r̂` de la sortie de chaque
     composant, à chaque couche et position : `x' = x − (x · r̂) r̂`. Réversible, idéal pour
     l'exploration et la mesure.
   - **Weight orthogonalization (permanente)** : pour chaque matrice `W` qui écrit dans le
     residual stream (out_proj de l'attention, down_proj du MLP, et embeddings),
     `W' = W − r̂ r̂ᵀ W`. Le modèle perd la capacité d'écrire dans la direction de refus.
     C'est ce qui produit un modèle de poids redistribuable.

## Trois residual streams cibles

Dans une architecture decoder-only de type Llama, on peut mesurer/intervenir à trois
endroits par bloc : **pre** (début de bloc), **mid** (entre attention et MLP), **post**
(après MLP). Le papier original mesurait les trois ; beaucoup d'implémentations récentes
basées sur `transformers` accèdent à l'équivalent de **post** via `hidden_states`.

## Les variantes (à connaître pour choisir)

| Variante | Idée clé | Quand l'utiliser |
|---|---|---|
| **Conventionnelle** | Soustraction directe de `r̂` | Baseline, modèles simples |
| **Projected** (grimjim, oct. 2025) | Orthogonaliser la direction de refus contre la direction *harmless* avant ablation ; ne retirer que la composante mécaniquement pertinente | Quand la soustraction brute dégrade trop (ex. Gemma 3) |
| **Norm-Preserving Biprojected** (grimjim, nov. 2025) | Préserver la norme des poids après orthogonalisation + projeter entre couches | État de l'art "manuel" pour éviter la "lobotomie" |
| **Gabliteration** (Gülmez, arXiv:2512.18901, 2026) | Modification multi-directionnelle adaptative des poids | Altération comportementale sélective, multi-directions |

**Pourquoi "projected" et "norm-preserving" comptent** : la soustraction brute altère la
*magnitude* des neurones, détruisant les normes de features apprises pendant
l'entraînement — c'est la cause classique de dégradation (logique cassée, hallucinations).
Fondement théorique : Zhao et al. (2025) « LLMs Encode Harmfulness and Refusal Separately » —
refus et nocivité sont encodés séparément, donc on peut retirer le refus en touchant moins
au reste.

Détails maths complets de chaque variante : voir `references/variants_math.md`.

## Le paysage des outils (ne pas réinventer la roue)

Avant de coder un pipeline from scratch, situer le projet par rapport à l'existant. Détails
et liens dans `references/tooling_landscape.md`.

- **Heretic** (`p-e-w/heretic`, `pip install heretic-llm`) — entièrement automatique,
  optimiseur TPE/Optuna qui co-minimise le nombre de refus *et* la divergence KL vis-à-vis
  du modèle original. La référence du domaine (>3000 modèles publiés). Supporte dense, MoE,
  certains hybrides (Qwen3).
- **Abliterix** (`wuwangzhang1216/abliterix`) — dérivé de Heretic ; ablation
  *expert-granular* pour les MoE, LoRA, 150+ configs préfaites, et le benchmark
  *HonestAbliterationBench*.
- **llm-abliteration** (`jim-plus/llm-abliteration`, miroir `NousResearch`) — implémentation
  manuelle avec flags `--projected`, `--normpreserve`, `--invert` ; permet d'explorer chaque
  option indépendamment.
- **OBLITERATUS** (`elder-plinius/OBLITERATUS`) — cross-hardware/model/method, gère Conv1D et
  Linear, archis custom via `trust_remote_code`.
- **ErisForge / DECCP** — wrappers runtime / single-pass quantifié.

**Recommandation par défaut** : pour un nouveau projet, partir de Heretic (ou d'un fork) et
y ajouter de la valeur, plutôt que de réimplémenter la collecte d'activations + l'optimiseur.
Réimplémenter from scratch n'a de sens que comme exercice d'apprentissage ou si on cible une
architecture/contrainte non couverte.

## Décisions à arrêter au démarrage d'un projet

Quand l'utilisateur démarre un logiciel d'abliteration, clarifier :
1. **Réutiliser un outil existant** (fork/wrapper de Heretic) **vs from scratch** ? Cela
   change tout le reste.
2. **Architectures cibles** : dense uniquement, ou MoE / hybride / multimodal ? (impacte
   fortement le pipeline — voir `abliteration-pipeline`).
3. **Modalité** : hooks inference-time (exploration) et/ou orthogonalisation permanente
   (livraison de poids) ?
4. **Variante** : conventionnelle pour démarrer, puis projected / norm-preserving si
   dégradation.
5. **Critère de succès chiffré** : taux de refus cible + budget de divergence KL /
   rétention de benchmark (voir `abliteration-eval`).

## Cadre responsable

L'abliteration est dual-use : elle sert l'interprétabilité, l'étude de la robustesse de
l'alignement, et la réduction des *sur-refus* (faux positifs sur requêtes bénignes), mais
produit aussi des modèles capables de générer du contenu nuisible. Pour tout projet,
documenter l'usage visé, et garder en tête la recherche défensive associée — p. ex. Abu
Shairah et al. (KAUST, mai 2025) « An Embarrassingly Simple Defense Against LLM Abliteration
Attacks » (extended-refusal fine-tuning). Ne jamais ajouter à un pipeline d'abliteration de
fonctionnalité dont le but serait de produire des catégories de contenu gravement
dangereuses ; rester sur l'outil de modification de modèle générique.
