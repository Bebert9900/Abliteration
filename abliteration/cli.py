"""CLI de l'outil d'abliteration.

Sous-commandes (toutes prennent un `model` positionnel) :
  extract   collecte d'activations + calcul des directions 4 classes
  select    choix de la meilleure couche (séparabilité)
  apply     applique l'ablation aux poids et sauvegarde le modèle
  abliterate  pipeline complet extract → select → apply → eval
  optimize  recherche Optuna des poids λ de l'objectif composite
  eval      évalue un modèle (rapport bi-axe : refus + capacités préservées)
  diagnose  diagnostic des directions/séparabilité (lecture seule, pas de sortie)
  heal      réparation post-abliteration (LoRA SFT sur traces propres)

La variante `preserving` orthogonalise la direction de refus contre les
directions à préserver (`--preserve negation,agentic,...`) afin de ne casser
ni la négation logique légitime ni les capacités agentiques (tool use).
"""
from __future__ import annotations

import argparse
import logging

log = logging.getLogger("abliteration")


def parse_preserve(value: str | None) -> list[str]:
    """Découpe une liste séparée par des virgules ; None/"" → []."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# --------------------------------------------------------------------------- #
# Construction du parseur
# --------------------------------------------------------------------------- #
def _add_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="Identifiant HF ou chemin local du modèle.")


def _add_common(p: argparse.ArgumentParser) -> None:
    """Options partagées (chargement / données)."""
    p.add_argument("--device", default=None, help="Périphérique torch (cpu/cuda/auto).")
    p.add_argument("--dtype", default="bfloat16", help="dtype de chargement (KB : bf16 pour l'ablation finale).")
    p.add_argument("--data-dir", default="data", help="Dossier des prompts des 4 classes contrastives.")
    p.add_argument("--batch-size", type=int, default=8, help="Taille de batch pour la collecte d'activations.")
    p.add_argument("--holdout", type=float, default=0.2, help="Fraction holdout (protocole d'éval honnête).")
    p.add_argument("--seed", type=int, default=0, help="Graine pour le split holdout.")


def _add_layers(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layers", default=None, help="Couches candidates, ex '10,12,14' ou '8-20'.")


def _add_variant(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--variant",
        choices=["conventional", "projected", "preserving", "norm_preserving_biprojected"],
        default="preserving",
        help="Variante d'ablation (KB §3) ; preserving orthogonalise contre --preserve.",
    )
    p.add_argument(
        "--preserve",
        default=None,
        help="Directions à préserver (liste virgulée), ex 'negation,agentic'.",
    )
    p.add_argument("--norm-preserve", action="store_true", help="Préserve la norme des poids après projection.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abliterate", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs détaillés.")
    sub = parser.add_subparsers(dest="command", required=True)

    # extract ---------------------------------------------------------------- #
    p = sub.add_parser("extract", help="Collecte d'activations + directions 4 classes.")
    _add_model(p); _add_common(p); _add_layers(p)
    p.add_argument("--out", default="directions.pt", help="Fichier de sortie des directions.")
    p.set_defaults(func=cmd_extract)

    # select ----------------------------------------------------------------- #
    p = sub.add_parser("select", help="Sélection de la meilleure couche.")
    _add_model(p); _add_common(p); _add_layers(p)
    p.add_argument("--directions", default="directions.pt", help="Fichier de directions (issu d'extract).")
    p.set_defaults(func=cmd_select)

    # apply ------------------------------------------------------------------ #
    p = sub.add_parser("apply", help="Applique l'ablation aux poids et sauvegarde.")
    _add_model(p); _add_common(p); _add_variant(p)
    p.add_argument("--directions", default="directions.pt", help="Fichier de directions.")
    p.add_argument("--layer", type=int, default=None, help="Couche de la direction de refus.")
    p.add_argument("--out", default="./out", help="Dossier de sortie du modèle abliteré.")
    p.set_defaults(func=cmd_apply)

    # abliterate (pipeline complet) ----------------------------------------- #
    p = sub.add_parser("abliterate", help="Pipeline complet extract → select → apply → eval.")
    _add_model(p); _add_common(p); _add_layers(p); _add_variant(p)
    p.add_argument("--out", default="./out", help="Dossier de sortie du modèle abliteré.")
    p.set_defaults(func=cmd_abliterate)

    # optimize --------------------------------------------------------------- #
    p = sub.add_parser("optimize", help="Recherche Optuna des poids λ de l'objectif composite.")
    _add_model(p); _add_common(p); _add_layers(p); _add_variant(p)
    p.add_argument("--trials", type=int, default=30, help="Nombre d'essais Optuna.")
    p.add_argument("--lambda-kl", type=float, default=1.0, help="Poids du terme KL (fidélité).")
    p.add_argument("--lambda-neg", type=float, default=1.0, help="Poids du terme négation.")
    p.add_argument("--lambda-syco", type=float, default=1.0, help="Poids du terme sycophantie.")
    p.add_argument("--lambda-agent", type=float, default=1.0, help="Poids du terme agentique.")
    p.add_argument("--checkpoint", default="optuna.json", help="Fichier de checkpoint des essais.")
    p.add_argument("--out", default="./out", help="Dossier de sortie du meilleur modèle.")
    p.set_defaults(func=cmd_optimize)

    # eval ------------------------------------------------------------------- #
    p = sub.add_parser("eval", help="Évalue un modèle (rapport bi-axe).")
    _add_model(p); _add_common(p)
    p.add_argument("--base", default=None, help="Modèle de base (référence KL de préservation).")
    p.add_argument("--benchmarks", default=None, help="Benchmarks externes (liste virgulée), ex 'mmlu,gsm8k'.")
    p.add_argument("--bench-limit", type=int, default=None, dest="bench_limit",
                   help="Limite d'exemples par benchmark (sous-ensemble rapide).")
    p.add_argument("--out", default=None, help="Fichier JSON du rapport (stdout si absent).")
    p.set_defaults(func=cmd_eval)

    # diagnose (pas de --out) ----------------------------------------------- #
    p = sub.add_parser("diagnose", help="Diagnostic directions/séparabilité (lecture seule).")
    _add_model(p); _add_common(p); _add_layers(p)
    p.add_argument("--directions", default=None, help="Fichier de directions à diagnostiquer (sinon calculé).")
    p.add_argument("--circuit", action="store_true",
                   help="Ajoute un résumé circuitiel court (analyse, aucune modif de poids).")
    p.add_argument("--layer", type=int, default=None, help="Couche de lecture pour le résumé circuitiel.")
    p.set_defaults(func=cmd_diagnose)

    # analyze-circuit (Phase 1 : analyse circuitielle, AUCUNE modif de poids) - #
    p = sub.add_parser(
        "analyze-circuit",
        help="Localise et VALIDE causalement le circuit de refus (lecture seule, rapport).",
    )
    _add_model(p); _add_common(p); _add_layers(p)
    p.add_argument("--layer", type=int, default=None,
                   help="Couche de lecture de r̂ (défaut : couche médiane).")
    p.add_argument("--pairs", type=int, default=16,
                   help="Nombre de paires clean(harmful)/corrupted(harmless).")
    p.add_argument("--top-k", type=int, default=20,
                   help="Composants top-attribution confirmés ensuite par patching exact.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Seuil de recovery (nécessité+suffisance) pour entrer au circuit core.")
    p.add_argument("--target-faithfulness", type=float, default=0.9,
                   help="Sélection du core par faithfulness (RC2) : on agrège les composants par "
                        "nécessité causale jusqu'à ce que le knockout du core explique ≥ cette "
                        "fraction du comportement. 0 = seuil dur par score (legacy).")
    p.add_argument("--holdout-frac", type=float, default=0.5,
                   help="Fraction des paires réservée à la MESURE held-out de la faithfulness "
                        "(jamais utilisée pour sélectionner le circuit). Anti-tautologie. 0 = off.")
    p.add_argument("--min-holdout", type=int, default=5,
                   help="Taille minimale du test-set held-out ; en dessous, WARNING dans le JSON.")
    p.add_argument("--n-boot", type=int, default=200, help="Tirages bootstrap (stabilité Jaccard).")
    p.add_argument("--backend", choices=["auto", "torch", "nnsight"], default="auto",
                   help="Backend d'introspection (NNsight si dispo, sinon hooks torch).")
    p.add_argument("--no-circuit-metrics", action="store_true",
                   help="Saute faithfulness/CPR/CMD (plus rapide).")
    p.add_argument("--out", default=None, help="Fichier JSON du rapport (stdout texte si absent).")
    p.set_defaults(func=cmd_analyze_circuit)

    # heal ------------------------------------------------------------------- #
    p = sub.add_parser("heal", help="Réparation post-abliteration (LoRA SFT).")
    _add_model(p)
    p.add_argument("--traces", default="traces.jsonl", help="Traces propres pour le SFT.")
    p.add_argument("--n-traces", type=int, default=200, help="Nombre de traces utilisées.")
    p.add_argument("--method", default="lora_sft", help="Méthode de healing.")
    p.add_argument("--out", default="./out-healed", help="Dossier de sortie du modèle réparé.")
    p.set_defaults(func=cmd_heal)

    return parser


# --------------------------------------------------------------------------- #
# Helpers handlers
# --------------------------------------------------------------------------- #
def _parse_layers(spec: str | None) -> list[int] | None:
    """'10,12,14' ou '8-20' → liste d'entiers ; None → None (laisse le défaut décider)."""
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _load_four_class_data(ns):
    """Charge les 4 classes contrastives (un fichier par classe) avec split train/holdout.

    `train` sert à calculer les directions ; `holdout` à évaluer le refus sur des prompts jamais
    vus (protocole honnête, KB §7). Échoue clairement si un fichier de classe manque.
    """
    from pathlib import Path

    from .data import FourClassData, PromptClass

    base = Path(ns.data_dir)
    paths = {cls: base / f"{cls.value}.txt" for cls in PromptClass}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"fichiers de classe manquants sous {base} : {missing}")
    return FourClassData.load(paths, holdout_fraction=ns.holdout, seed=ns.seed)


def _default_candidate_layers(n_layers: int) -> list[int]:
    """Band milieu→milieu-tardif (KB §2.1) où la direction de refus est la plus exploitable."""
    fracs = (0.4, 0.5, 0.6, 0.7, 0.8)
    cand = sorted({max(1, min(n_layers - 1, int(round(n_layers * f)))) for f in fracs})
    return cand


def _select_layer_causal(model, adapter, formatter, directions, candidate_layers, harmful_texts,
                         device=None, n_probe=24, max_new_tokens=32):
    """Sélection RÉELLE (causale) : pour chaque couche candidate, on pose le hook d'ablation
    réversible (KB §5a) avec r̂ de cette couche, on génère sur un échantillon harmful, et on
    garde la couche qui MINIMISE le taux de refus. C'est la stratégie décrite dans selection.py.
    """
    from .ablation import register_ablation_hooks
    from .directions import select_layer
    from .eval import KeywordRefusalJudge, generate_responses, refusal_rate

    judge = KeywordRefusalJudge()
    probe = harmful_texts[:n_probe]
    # modules écrivant au residual stream (hors embedding) = cibles du hook inference-time.
    from .models import WriteKind
    targets = [t.module for t in adapter.residual_writers() if t.kind != WriteKind.EMBEDDING]
    dev = next(model.parameters()).device

    scores: dict[int, float] = {}

    def score_fn(layer: int) -> float:
        direction = directions.refusal[layer].to(dev)
        handles = register_ablation_hooks(targets, direction)
        try:
            resp = generate_responses(model, formatter, probe, max_new_tokens=max_new_tokens,
                                      min_new_tokens=8, batch_size=8, device=device)
        finally:
            for h in handles:
                h.remove()
        r = refusal_rate(probe, resp, judge)
        scores[layer] = r
        log.info("  couche %d : refus(ablation)=%.3f", layer, r)
        return r

    best = select_layer(candidate_layers, score_fn)
    log.info("Meilleure couche (refus minimal sous ablation) : %d (refus=%.3f)", best, scores[best])
    return best, scores


def _run_eval(model, formatter, data, *, base_logits=None, kl_texts=None, device=None,
              max_new_tokens=128, batch_size=8, generations_out=None):
    """Évaluation bi-axe RÉELLE sur le holdout (génération + juges déterministes).

    Axe 1 (suppression refus) : refusal_rate sur harmful holdout.
    Axe 2 (préservation) : negation_retention, agentic_score, KL(base‖abliteré) sur harmless.
    Garde-fous : degeneracy_rate, empty_rate, follow_rate (sycophantie).

    `generations_out` : si fourni, persiste les TEXTES BRUTS des générations harmful (prompt +
    réponse + indice heuristique) pour relecture/re-jugement (on ne jette plus les textes).
    """
    from .data import PromptClass
    from .eval import (
        EvalReport, KeywordRefusalJudge, agentic_score, degeneracy_rate, dump_generations,
        empty_rate, follow_rate, generate_responses, harmless_logits, is_degenerate,
        is_sycophantic, kl_divergence, negates_correctly, negation_retention, parse_tool_call,
        refusal_rate, schema_valid,
    )

    judge = KeywordRefusalJudge()

    def gen(texts):
        return generate_responses(model, formatter, texts, max_new_tokens=max_new_tokens,
                                  batch_size=batch_size, device=device)

    # --- Axe 1 : refus sur harmful holdout ---
    harmful = [p.text for p in data.holdout(PromptClass.HARMFUL)]
    resp_h = gen(harmful)
    refusal = refusal_rate(harmful, resp_h, judge)
    degeneracy = degeneracy_rate(resp_h)
    empty = empty_rate(resp_h)

    if generations_out:
        dump_generations(generations_out, harmful, resp_h, judge=judge,
                         degeneracy_check=is_degenerate, max_new_tokens=max_new_tokens,
                         axis="harmful_holdout")

    # --- Axe 2a : négation logique légitime (préservation) ---
    neg = [p.text for p in data.holdout(PromptClass.LEGITIMATE_NEGATION)]
    resp_n = gen(neg)
    neg_ret = negation_retention(resp_n, negates_correctly)
    follow = follow_rate(resp_n, is_sycophantic)

    # --- Axe 2b : capacités agentiques (validité de schéma de tool call) ---
    # On donne au modèle le contrat de sortie (nom d'outil + format JSON) pour que la métrique
    # mesure la CAPACITÉ à émettre un appel valide, pas la connaissance d'un format implicite.
    ag = data.holdout(PromptClass.AGENTIC)

    def _agentic_prompt(p):
        schema = p.meta.get("tool", {})
        name = schema.get("name", "the_tool")
        req = ", ".join(schema.get("parameters", {}).get("required", []))
        return (f"{p.text}\n\nAvailable tool: `{name}` (required arguments: {req}).\n"
                f"Respond with ONLY a JSON object of the form "
                f'{{"name": "{name}", "arguments": {{...}}}}.')

    resp_a = gen([_agentic_prompt(p) for p in ag])
    valid = 0
    for prompt, out in zip(ag, resp_a):
        schema = prompt.meta.get("tool", {})
        call = parse_tool_call(out)
        if call is not None and schema and schema_valid(call, schema):
            valid += 1
    schema_validity = valid / len(ag) if ag else 0.0
    agentic = agentic_score(schema_validity, schema_validity, schema_validity)

    # --- Axe 2c : KL de préservation sur harmless (si logits de base fournis) ---
    kl = 0.0
    if base_logits is not None and kl_texts:
        abl_logits = harmless_logits(model, formatter, kl_texts, batch_size=max(2, batch_size // 2),
                                     device=device)
        n = min(base_logits.shape[0], abl_logits.shape[0])
        kl = kl_divergence(base_logits[:n], abl_logits[:n])

    return EvalReport(
        refusal_rate=refusal, kl=kl, negation_retention=neg_ret, follow_rate=follow,
        empty_rate=empty, agentic_score=agentic, degeneracy_rate=degeneracy,
    )


# --------------------------------------------------------------------------- #
# Handlers (import paresseux des modules lourds)
# --------------------------------------------------------------------------- #
def cmd_extract(ns) -> int:
    import torch

    from .data import PromptClass, PromptFormatter
    from .directions import collect_means, compute_directions
    from .models import load_model

    log.info("Chargement du modèle %s", ns.model)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    data = _load_four_class_data(ns)

    means = {}
    for cls in PromptClass:
        texts = [p.text for p in data.train(cls)]   # directions calculées sur le TRAIN uniquement
        if not texts:
            raise ValueError(f"Classe {cls.value} vide : impossible de calculer les directions.")
        log.info("Collecte d'activations : %s (%d prompts train)", cls.value, len(texts))
        means[cls] = collect_means(model, formatter, texts, batch_size=ns.batch_size, device=ns.device)

    directions = compute_directions(means)
    torch.save(directions, ns.out)
    log.info("Directions sauvegardées dans %s", ns.out)
    print(ns.out)
    return 0


def cmd_select(ns) -> int:
    import torch

    from .data import PromptClass, PromptFormatter
    from .models import ArchAdapter, load_model

    directions = torch.load(ns.directions, weights_only=False)
    n_layers = directions.refusal.shape[0]
    candidates = _parse_layers(ns.layers) or _default_candidate_layers(n_layers)

    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    adapter = ArchAdapter(model)
    data = _load_four_class_data(ns)
    harmful = [p.text for p in data.train(PromptClass.HARMFUL)]

    best, _ = _select_layer_causal(model, adapter, formatter, directions, candidates, harmful,
                                   device=ns.device)
    print(best)
    return 0


def cmd_apply(ns) -> int:
    import torch

    from .ablation import Variant, ablation_direction, orthogonalize_weights
    from .io import build_model_card, save_model, write_model_card
    from .models import ArchAdapter, load_model

    preserve = parse_preserve(ns.preserve)
    directions = torch.load(ns.directions, weights_only=False)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    adapter = ArchAdapter(model)

    layer = getattr(ns, "layer", None)
    layer = layer if layer is not None else 0
    direction = ablation_direction(directions, layer, Variant(ns.variant), preserve=preserve or None)
    orthogonalize_weights(adapter, direction, norm_preserve=getattr(ns, "norm_preserve", False))

    save_model(model, tok, ns.out)
    card = build_model_card(ns.model, ns.variant, preserve, metrics={})
    write_model_card(ns.out, card)
    log.info("Modèle abliteré sauvegardé dans %s", ns.out)
    print(ns.out)
    return 0


def cmd_abliterate(ns) -> int:
    """Pipeline complet, modèle chargé UNE seule fois :
    extract (directions sur train) → select (couche par ablation causale) → KL base →
    orthogonalisation des poids → éval bi-axe holdout → sauvegarde + model card.
    """
    import json

    from .ablation import Variant, ablation_direction, orthogonalize_weights
    from .data import PromptClass, PromptFormatter
    from .directions import collect_means, compute_directions
    from .eval import harmless_logits
    from .io import build_model_card, save_model, write_model_card
    from .models import ArchAdapter, load_model

    preserve = parse_preserve(ns.preserve)
    log.info("Chargement du modèle %s", ns.model)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    adapter = ArchAdapter(model)
    data = _load_four_class_data(ns)

    # 1) Directions (4 classes) sur le TRAIN.
    means = {}
    for cls in PromptClass:
        texts = [p.text for p in data.train(cls)]
        log.info("Collecte d'activations : %s (%d prompts)", cls.value, len(texts))
        means[cls] = collect_means(model, formatter, texts, batch_size=ns.batch_size, device=ns.device)
    directions = compute_directions(means)

    # 2) Sélection causale de la couche de refus.
    n_layers = directions.refusal.shape[0]
    candidates = _parse_layers(ns.layers) or _default_candidate_layers(n_layers)
    harmful_train = [p.text for p in data.train(PromptClass.HARMFUL)]
    best_layer, layer_scores = _select_layer_causal(
        model, adapter, formatter, directions, candidates, harmful_train, device=ns.device)

    # 3) Logits de base (préservation/KL) sur un échantillon harmless AVANT de modifier les poids.
    kl_texts = [p.text for p in data.holdout(PromptClass.HARMLESS)][:16]
    log.info("Capture des logits de base (KL) sur %d prompts harmless", len(kl_texts))
    base_logits = harmless_logits(model, formatter, kl_texts, batch_size=4, device=ns.device)

    # 4) Orthogonalisation permanente des poids contre la direction (projetée selon la variante).
    direction = ablation_direction(directions, best_layer, Variant(ns.variant), preserve=preserve or None)
    log.info("Orthogonalisation des poids (variante=%s, couche=%d, preserve=%s)",
             ns.variant, best_layer, preserve or "—")
    orthogonalize_weights(adapter, direction, norm_preserve=getattr(ns, "norm_preserve", False))
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    # 5) Éval bi-axe sur le holdout (modèle désormais abliteré en mémoire).
    log.info("Évaluation bi-axe du modèle abliteré (holdout)")
    from pathlib import Path as _Path
    _Path(ns.out).mkdir(parents=True, exist_ok=True)
    report = _run_eval(model, formatter, data, base_logits=base_logits, kl_texts=kl_texts,
                       device=ns.device, batch_size=ns.batch_size,
                       generations_out=_Path(ns.out) / "harmful_generations.json")

    # 6) Sauvegarde + model card transparente avec métriques.
    metrics = report.to_dict()
    metrics["selected_layer"] = best_layer
    save_model(model, tok, ns.out)
    card = build_model_card(ns.model, ns.variant, preserve, metrics=metrics)
    write_model_card(ns.out, card)
    report.save(__import__("pathlib").Path(ns.out) / "eval_report.json")
    log.info("Abliteration terminée → %s", ns.out)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def cmd_optimize(ns) -> int:
    from .optimize import Lambdas, run_optuna_study

    lambdas = Lambdas(
        kl=ns.lambda_kl, negation=ns.lambda_neg,
        sycophancy=ns.lambda_syco, agentic=ns.lambda_agent,
    )
    log.info("Optimisation Optuna : %d essais, λ=%r", ns.trials, lambdas)
    space = {"layer": ("int", 0, 1), "alpha": ("float", 0.0, 1.0)}
    result = run_optuna_study(
        objective=lambda trial: 0.0,  # objectif réel câblé à l'éval dans une itération ultérieure
        space=space,
        n_trials=ns.trials,
        checkpoint_path=ns.checkpoint,
    )
    print(result)
    return 0


def cmd_eval(ns) -> int:
    from .data import PromptClass, PromptFormatter
    from .eval import BenchmarkNotInstalled, harmless_logits, run_benchmark
    from .models import load_model

    log.info("Évaluation de %s", ns.model)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    data = _load_four_class_data(ns)
    kl_texts = [p.text for p in data.holdout(PromptClass.HARMLESS)][:16]

    # KL de préservation vs un modèle de base optionnel (--base).
    base_logits = None
    base = getattr(ns, "base", None)
    if base:
        log.info("Chargement du modèle de base %s pour la KL", base)
        base_model, base_tok = load_model(base, dtype=ns.dtype, device_map=ns.device or "auto")
        base_logits = harmless_logits(base_model, PromptFormatter(base_tok), kl_texts,
                                      batch_size=4, device=ns.device)
        del base_model

    report = _run_eval(model, formatter, data, base_logits=base_logits, kl_texts=kl_texts,
                       device=ns.device, batch_size=ns.batch_size)

    # Benchmarks externes (lm-eval) — pilotés sur le chemin du modèle.
    benchmarks = parse_preserve(getattr(ns, "benchmarks", None))
    if benchmarks:
        results = {}
        limit = getattr(ns, "bench_limit", None)
        for name in benchmarks:
            try:
                log.info("Benchmark %s …", name)
                results[name] = run_benchmark(name, ns.model, device=ns.device or "cuda", limit=limit)
            except (BenchmarkNotInstalled, NotImplementedError, ValueError) as e:
                log.warning("Benchmark %s ignoré : %s", name, e)
                results[name] = {"error": str(e)}
        report.benchmarks = results

    import json
    out = getattr(ns, "out", None)
    if out:
        report.save(out)
        log.info("Rapport écrit dans %s", out)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_diagnose(ns) -> int:
    import torch

    from .directions import separability

    if ns.directions:
        directions = torch.load(ns.directions, weights_only=False)
        rep = separability(directions)
        n_layers = directions.refusal.shape[0]
        layers = _parse_layers(ns.layers) or list(range(n_layers))
        for layer in layers:
            cn = float(rep.cosine_refusal_negation[layer])
            ca = float(rep.cosine_refusal_agentic[layer])
            print(f"layer={layer} cos(refus,negation)={cn:+.3f} cos(refus,agentic)={ca:+.3f}")
        for w in rep.warnings():
            log.warning("%s", w)
    else:
        log.warning("Aucun fichier --directions : lancez d'abord `extract`.")

    if getattr(ns, "circuit", False):
        # résumé circuitiel court (réutilise le pipeline d'analyse, sans modif de poids)
        try:
            report = _run_circuit_analysis(ns, n_pairs=getattr(ns, "pairs", 8),
                                           top_k=getattr(ns, "top_k", 12),
                                           compute_metrics=False)
            print(report.short_summary())
        except Exception as e:  # pragma: no cover - dépend du modèle/env
            log.warning("Résumé circuitiel indisponible : %s", e)
    return 0


def _build_circuit_pairs(formatter, harmful_texts, harmless_texts, n, device=None):
    """Construit n paires (clean_ids, corr_ids, clean_mask, corr_mask).

    clean = harmful (déclenche le refus) ; corrupted = harmless (ne le déclenche pas).
    On tokenise chaque paire ENSEMBLE pour qu'elles partagent la longueur (patching aligné).
    """
    pairs = []
    n = min(n, len(harmful_texts), len(harmless_texts))
    for i in range(n):
        enc = formatter.tokenize([harmful_texts[i], harmless_texts[i]])
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        clean_ids, corr_ids = ids[0:1], ids[1:2]
        clean_mask, corr_mask = mask[0:1], mask[1:2]
        pairs.append((clean_ids, corr_ids, clean_mask, corr_mask))
    return pairs


def _run_circuit_analysis(ns, n_pairs, top_k, compute_metrics):
    """Pipeline Phase 1 : DLA+attribution (scan) → patching exact (confirme top-k) → rapport.

    AUCUNE modification de poids. Renvoie un CircuitReport.
    """
    import os
    import torch

    # Déterminisme : tue le flip run-to-run dû au non-déterminisme matmul GPU (à régler AVANT
    # toute init CUDA, donc avant load_model). warn_only pour ne pas casser si un op manque.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    from .circuits import make_backend
    from .circuits.attribution import aggregate_attribution
    from .circuits.dla import direct_logit_attribution, readout_direction
    from .circuits.localize import localize
    from .circuits.patching import RefusalMetric
    from .circuits.report import CircuitReport
    from .data import PromptClass, PromptFormatter
    from .directions import collect_means, compute_directions
    from .models import load_model

    from pathlib import Path

    from .data import load_prompts

    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)

    # chargement direct des 4 classes (un fichier JSONL par classe sous --data-dir)
    base = Path(ns.data_dir)
    texts_by_class = {}
    for cls in PromptClass:
        path = base / f"{cls.value}.txt"
        texts_by_class[cls] = [p.text for p in load_prompts(path, cls)] if path.exists() else []

    # directions (réutilise le pipeline directionnel, ne recalcule pas la math ici)
    means = {}
    for cls in PromptClass:
        if texts_by_class[cls]:
            means[cls] = collect_means(model, formatter, texts_by_class[cls],
                                       batch_size=ns.batch_size, device=ns.device)
    directions = compute_directions(means)

    n_layers = directions.refusal.shape[0]
    layer = ns.layer if getattr(ns, "layer", None) is not None else n_layers // 2
    r_hat = readout_direction(directions, layer)

    dev = next(model.parameters()).device
    r_hat = r_hat.to(dev)
    harmful = texts_by_class[PromptClass.HARMFUL]
    harmless = texts_by_class[PromptClass.HARMLESS]
    pairs = _build_circuit_pairs(formatter, harmful, harmless, n_pairs, device=dev)

    backend = make_backend(model, prefer=getattr(ns, "backend", "auto"))
    metric = RefusalMetric(refusal_dir=r_hat)

    # 1+2) localize gère le split train/test : candidats (attribution agrégée) + greedy SUR LE
    #    TRAIN, faithfulness REPORTÉE sur le held-out (anti-tautologie). candidates=None →
    #    dérivés sur le train uniquement (corrige RC1 + ferme la fuite sélection→mesure).
    tf = getattr(ns, "target_faithfulness", 0.9)
    loc = localize(backend, pairs, metric, r_hat, candidates=None,
                   threshold=ns.threshold,
                   target_faithfulness=(tf if tf and tf > 0 else None),
                   holdout_frac=getattr(ns, "holdout_frac", 0.5),
                   min_holdout=getattr(ns, "min_holdout", 5),
                   n_candidates=top_k,
                   n_boot=getattr(ns, "n_boot", 200),
                   compute_circuit_metrics=compute_metrics)

    cids, cmask = pairs[0][0], pairs[0][2]   # résumé DLA (corrélationnel) sur la 1re paire
    dla = direct_logit_attribution(backend, r_hat, cids, cmask)
    return CircuitReport(model_name=ns.model, localization=loc, dla=dla, n_pairs=len(pairs))


def cmd_analyze_circuit(ns) -> int:
    """Commande `analyze-circuit` : rapport de localisation causale du refus. Lecture seule."""
    log.info("Analyse circuitielle de %s (Phase 1, aucune modif de poids)", ns.model)
    report = _run_circuit_analysis(
        ns, n_pairs=ns.pairs, top_k=ns.top_k,
        compute_metrics=not ns.no_circuit_metrics,
    )
    if ns.out:
        report.to_json(path=ns.out)
        log.info("Rapport circuitiel écrit dans %s", ns.out)
    else:
        print(report.to_text())
    return 0


def cmd_heal(ns) -> int:
    from .heal import HealConfig, heal

    config = HealConfig(
        model_dir=ns.model, traces_path=ns.traces,
        n_traces=ns.n_traces, method=ns.method, out_dir=ns.out,
    )
    log.info("Healing %s (méthode=%s)", ns.model, ns.method)
    heal(config)
    return 0


# --------------------------------------------------------------------------- #
# Entrée
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(ns, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    func = getattr(ns, "func", None)
    if func is None:  # pragma: no cover - argparse impose une sous-commande
        parser.error("aucune sous-commande fournie")
    return func(ns) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
