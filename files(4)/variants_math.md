# Maths des variantes d'abliteration

Notations : `r` direction de refus brute, `r̂` sa version unitaire, `h` direction *harmless*,
`ĥ` unitaire. `W` matrice d'un composant écrivant dans le residual stream. `x` vecteur
d'activation.

## 1. Conventionnelle

Direction : `r_l = μ_harmful_l − μ_harmless_l`, `r̂_l = r_l / ‖r_l‖`.

Ablation inference-time (hook sur chaque écriture au residual stream) :
```
x' = x − (x · r̂) r̂
```

Orthogonalisation de poids (permanente), pour chaque matrice `W` qui écrit dans le residual
stream :
```
W' = W − r̂ (r̂ᵀ W)
```
Appliquer à : `o_proj` (attention), `down_proj` (MLP), et la matrice d'embedding. Effet : la
sortie de ces composants n'a plus de composante le long de `r̂`.

Limite : modifie aussi la magnitude effective des colonnes de `W`, ce qui peut dégrader des
features non liées au refus → "lobotomie".

## 2. Projected (grimjim, oct. 2025)

Hypothèse (Zhao et al. 2025) : refus et nocivité sont encodés séparément. La direction de
refus brute mélange « pousser vers le refus » et « pousser hors de la compliance ». Pour
l'abliteration, on ne veut retirer que la première composante.

On orthogonalise `r̂` contre la direction *harmless* `ĥ` avant ablation :
```
r̂_proj = normalize( r̂ − (r̂ · ĥ) ĥ )
```
puis on ablate avec `r̂_proj` au lieu de `r̂`. Cela évite de « casser » la direction de
compliance déjà optimisée par le fine-tuning d'instruction (retirer un *push away from
compliance* peut induire de la sur-compliance / hallucination).

Flag de référence dans `jim-plus/llm-abliteration` : `--projected` (orthogonalisation lors
de la *mesure*).

## 3. Norm-Preserving (grimjim, nov. 2025)

Après orthogonalisation, on réimpose la norme d'origine pour ne garder que la composante
*directionnelle* de l'intervention, sans effet de magnitude. Schématiquement, pour chaque
colonne `w` de `W` transformée en `w'` :
```
w'' = w' · (‖w‖ / ‖w'‖)
```
Résultat : on isole le fait que **la direction seule** est critique pour le refus, sans
emporter l'information de magnitude (souvent porteuse d'information comportementale
importante — cf. outliers de forte magnitude dans Gemma 3 12B). Flag : `--normpreserve`.

Remarque : certains modèles réagissent mal à la préservation de norme — à tester.

## 4. Biprojected (grimjim, nov. 2025)

Extension : on mesure la direction de refus sur une couche et on l'ablate sur une autre, en
projetant correctement d'une couche à l'autre. But : éviter les artefacts dus à des
directions trop spécifiques à une seule couche. Combiné avec norm-preserving → "Norm-
Preserving Biprojected Abliteration", utilisé en production par ArliAI (ex. GLM-4.6-
Derestricted) pour préserver le raisonnement.

## 5. Gabliteration (Gülmez, arXiv:2512.18901, 2026)

« Adaptive Multi-Directional Neural Weight Modification ». Au lieu d'une seule direction, un
sous-espace de plusieurs directions, avec pondération adaptative par couche/composant, pour
une altération comportementale plus sélective. Voir le papier pour la formulation exacte du
sous-espace et de l'adaptation.

## 6. Kernel de poids d'ablation (approche Heretic)

Plutôt qu'un poids d'ablation constant, Heretic paramètre un *kernel* sur les couches :
`max_weight`, `max_weight_position`, `min_weight`, `min_weight_distance` décrivent la forme
et la position de la fonction de poids d'ablation le long des couches, séparément par type de
composant (attention vs MLP). Plus `direction_index` (index d'une direction, ou `per layer`).
Ces paramètres sont ensuite optimisés (voir `abliteration-pipeline`, section optimisation).
