# Suite de skills Claude Code — Abliteration de LLM

Trois skills complémentaires pour un projet d'outil d'abliteration (directional ablation /
suppression de refus) de modèles de langage. Conçus pour Claude Code.

## Les trois skills

| Skill | Rôle | Se déclenche quand… |
|---|---|---|
| **abliteration-core** | Théorie, maths, catalogue des variantes et des outils | tu raisonnes sur l'algorithme, choisis une variante (conventionnelle / projected / norm-preserving biprojected / gabliteration) ou un outil à forker |
| **abliteration-pipeline** | Ingénierie : collecte d'activations, calcul de direction, orthogonalisation, archis (dense/MoE/hybride/VLM), quant, export GGUF | tu écris/débogues le code qui réalise l'abliteration |
| **abliteration-eval** | Mesure : taux de refus, divergence KL, benchmarks (MMLU/GSM8K), anti-gaming | tu juges si une abliteration a réussi sans casser le modèle |

Ils se renvoient l'un à l'autre : `core` (décider) → `pipeline` (construire) → `eval`
(vérifier), puis on itère.

## Installation dans Claude Code

Les skills vivent dans un dossier `skills/`, chaque skill dans son sous-dossier contenant un
`SKILL.md`. Deux portées :

- **Projet** (recommandé ici, committé dans le repo, partagé avec l'équipe) :
  `.claude/skills/<nom>/SKILL.md`
- **Perso** (dispo dans tous tes projets) : `~/.claude/skills/<nom>/SKILL.md`

```bash
# Depuis la racine de ton projet d'abliteration
mkdir -p .claude/skills
cp -r abliteration-core abliteration-pipeline abliteration-eval .claude/skills/
```

Vérifier que `SKILL.md` est bien à la racine de chaque sous-dossier (pas double-imbriqué).
Démarrer une nouvelle session, puis `/skills` pour confirmer le chargement. En cas de skill
non trouvé : nom exact `SKILL.md` (sensible à la casse), chemin
`.claude/skills/<nom>/SKILL.md`, et redémarrage de la session.

Doc officielle : https://code.claude.com/docs/en/skills

## Structure

```
abliteration-core/
├── SKILL.md
└── references/
    ├── variants_math.md       # maths de chaque variante
    └── tooling_landscape.md   # carte des outils + refs académiques
abliteration-pipeline/
└── SKILL.md
abliteration-eval/
└── SKILL.md
```

## Note

L'abliteration est une technique d'interprétabilité publiée (Arditi et al. 2024) et dual-use.
Ces skills incluent volontairement l'évaluation rigoureuse et le contexte de recherche
défensive. Documente l'usage visé de ton outil et publie des model cards transparentes.
