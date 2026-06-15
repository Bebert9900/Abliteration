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

from .output import emit_result

log = logging.getLogger("abliteration")


def parse_preserve(value: str | None) -> list[str]:
    """Découpe une liste séparée par des virgules ; None/"" → []."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# --------------------------------------------------------------------------- #
# Cache & reproductibilité
# --------------------------------------------------------------------------- #
def _cache_enabled(ns) -> bool:
    return not getattr(ns, "no_cache", False)


def _model_signature(ns, formatter):
    """Signature des entrées qui déterminent activations/logits (cf. cache.make_key)."""
    tok = formatter.tokenizer
    return (ns.model, ns.dtype, getattr(tok, "chat_template", "") or "")


def _cached_means(ns, model, formatter, texts, class_name):
    """collect_means avec cache disque (clé = modèle+dtype+template+classe+batch+prompts)."""
    from .cache import cached_tensor, make_key
    from .directions import collect_means
    if not texts:
        raise ValueError(f"Classe {class_name} vide : impossible de calculer les directions.")
    key = make_key("means", _model_signature(ns, formatter), class_name, ns.batch_size, tuple(texts))
    log.info("Collecte d'activations : %s (%d prompts train)", class_name, len(texts))
    return cached_tensor(
        key,
        lambda: collect_means(model, formatter, texts, batch_size=ns.batch_size,
                              device=ns.device).cpu(),
        enabled=_cache_enabled(ns), cache_dir=getattr(ns, "cache_dir", None),
    )


def _cached_base_logits(ns, model, formatter, texts, batch_size=4):
    """harmless_logits (logits de base pour la KL) avec cache disque."""
    from .cache import cached_tensor, make_key
    from .eval import harmless_logits
    if not texts:
        return None
    key = make_key("base_logits", _model_signature(ns, formatter), batch_size, tuple(texts))
    log.info("Capture des logits de base (KL) sur %d prompts harmless", len(texts))
    return cached_tensor(
        key,
        lambda: harmless_logits(model, formatter, texts, batch_size=batch_size, device=ns.device),
        enabled=_cache_enabled(ns), cache_dir=getattr(ns, "cache_dir", None),
    )


def _dump_run_config(ns, out_dir, extra=None):
    """Écrit run_config.json (traçabilité) : modèle, variante, hyperparamètres, hash des données."""
    import json
    from pathlib import Path

    from .cache import make_key

    cfg = {
        "model": ns.model,
        "dtype": getattr(ns, "dtype", None),
        "variant": getattr(ns, "variant", None),
        "preserve": parse_preserve(getattr(ns, "preserve", None)),
        "layers": getattr(ns, "layers", None),
        "holdout": getattr(ns, "holdout", None),
        "seed": getattr(ns, "seed", None),
        "batch_size": getattr(ns, "batch_size", None),
        "norm_preserve": getattr(ns, "norm_preserve", False),
        "data_dir": getattr(ns, "data_dir", None),
    }
    # Empreinte des fichiers de données : reproductibilité sans copier les prompts.
    data_dir = Path(getattr(ns, "data_dir", "data") or "data")
    sigs = {}
    for name in ("harmful", "harmless", "legitimate_negation", "agentic"):
        f = data_dir / f"{name}.txt"
        if f.exists():
            sigs[name] = make_key(f.read_text(encoding="utf-8"))
    cfg["data_hashes"] = sigs
    if extra:
        cfg.update(extra)
    path = Path(out_dir) / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return cfg


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
    p.add_argument("--no-cache", action="store_true", dest="no_cache",
                   help="Désactive le cache disque des activations/logits de base.")
    p.add_argument("--cache-dir", default=None, dest="cache_dir",
                   help="Dossier de cache (défaut : $ABLITERATION_CACHE ou ~/.cache/abliteration).")


def _add_layers(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layers", default=None, help="Couches candidates, ex '10,12,14' ou '8-20'.")


def _concept_choices() -> list[str]:
    """Concepts prédéfinis (pour les `choices` de --concept ; import léger, pas de torch)."""
    from .concepts import available_concepts
    return available_concepts()


def _add_variant(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--variant",
        choices=["conventional", "projected", "preserving", "norm_preserving_biprojected"],
        default="norm_preserving_biprojected",
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
    p.add_argument("--eval-limit", type=int, default=16, dest="eval_limit",
                   help="Nb de prompts/holdout évalués par essai (sous-échantillon rapide).")
    p.add_argument("--alpha-low", type=float, default=0.5, dest="alpha_low",
                   help="Borne basse de la force d'ablation graduée recherchée.")
    p.add_argument("--alpha-high", type=float, default=1.0, dest="alpha_high",
                   help="Borne haute de la force d'ablation graduée recherchée.")
    p.add_argument("--apply-best", action="store_true", dest="apply_best",
                   help="Grave le meilleur (couche, alpha) dans les poids et sauvegarde dans --out.")
    p.set_defaults(func=cmd_optimize)

    # eval ------------------------------------------------------------------- #
    p = sub.add_parser("eval", help="Évalue un modèle (rapport bi-axe).")
    _add_model(p); _add_common(p)
    p.add_argument("--base", default=None, help="Modèle de base (référence KL de préservation).")
    p.add_argument("--benchmarks", default=None, help="Benchmarks externes (liste virgulée), ex 'mmlu,gsm8k'.")
    p.add_argument("--bench-limit", type=int, default=None, dest="bench_limit",
                   help="Limite d'exemples par benchmark (sous-ensemble rapide).")
    p.add_argument("--kl-map", action="store_true", dest="kl_map",
                   help="Affiche le diagnostic KL par position (max/p95/top positions) ; requiert --base.")
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
    # Concept à localiser (défaut : refus via les 4 classes). Généralise l'analyse causale.
    p.add_argument("--concept", choices=_concept_choices(), default=None,
                   help="Concept prédéfini à localiser (défaut : refus).")
    p.add_argument("--pos", default=None, help="JSONL positif (concept ad hoc à localiser).")
    p.add_argument("--neg", default=None, help="JSONL négatif (concept ad hoc à localiser).")
    p.add_argument("--name", default=None, help="Nom du concept ad hoc à localiser.")
    p.set_defaults(func=cmd_analyze_circuit)

    # heal ------------------------------------------------------------------- #
    p = sub.add_parser("heal", help="Réparation post-abliteration (LoRA SFT).")
    _add_model(p)
    p.add_argument("--traces", default="traces.jsonl", help="Traces propres pour le SFT.")
    p.add_argument("--n-traces", type=int, default=200, help="Nombre de traces utilisées.")
    p.add_argument("--method", default="lora_sft", help="Méthode de healing.")
    p.add_argument("--out", default="./out-healed", help="Dossier de sortie du modèle réparé.")
    p.set_defaults(func=cmd_heal)

    # concept-direction (recherche : direction d'un concept arbitraire) ------ #
    p = sub.add_parser("concept-direction",
                       help="Direction d'un concept (registre ou ad hoc) par contraste.")
    _add_model(p); _add_common(p); _add_layers(p)
    p.add_argument("--concept", choices=_concept_choices(), default=None,
                   help="Concept prédéfini du registre.")
    p.add_argument("--pos", default=None, help="JSONL positif (concept ad hoc).")
    p.add_argument("--neg", default=None, help="JSONL négatif (concept ad hoc).")
    p.add_argument("--name", default=None, help="Nom du concept ad hoc.")
    p.add_argument("--out", default=None, help="Fichier de sortie de la direction (.pt).")
    p.set_defaults(func=cmd_concept_direction)

    # concept-probe (recherche : décodabilité linéaire couche par couche) ---- #
    p = sub.add_parser("concept-probe",
                       help="Sonde linéaire couche par couche : où un concept est-il décodable ?")
    _add_model(p); _add_common(p)
    p.add_argument("--concept", choices=_concept_choices(), default=None,
                   help="Concept prédéfini du registre.")
    p.add_argument("--pos", default=None, help="JSONL positif (concept ad hoc).")
    p.add_argument("--neg", default=None, help="JSONL négatif (concept ad hoc).")
    p.add_argument("--name", default=None, help="Nom du concept ad hoc.")
    p.set_defaults(func=cmd_concept_probe)

    # concept-steer (recherche : pilotage causal par ajout de direction) ----- #
    p = sub.add_parser("concept-steer",
                       help="Pilote la génération en ajoutant la direction d'un concept (steering).")
    _add_model(p); _add_common(p)
    p.add_argument("--concept", choices=_concept_choices(), default=None,
                   help="Concept prédéfini à injecter.")
    p.add_argument("--pos", default=None, help="JSONL positif (concept ad hoc).")
    p.add_argument("--neg", default=None, help="JSONL négatif (concept ad hoc).")
    p.add_argument("--name", default=None, help="Nom du concept ad hoc.")
    p.add_argument("--alpha", type=float, default=8.0, help="Intensité du pilotage.")
    p.add_argument("--layer", type=int, default=None, help="Couche de la direction (défaut : médiane).")
    p.add_argument("--preserve", default=None,
                   help="Concepts à préserver (orthogonalisation), ex 'negation,agentic'.")
    p.add_argument("--prompts", default=None,
                   help="JSONL de prompts à générer (défaut : holdout du concept).")
    p.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens")
    p.add_argument("--limit", type=int, default=8, help="Nb de prompts comparés (baseline/steered).")
    p.set_defaults(func=cmd_concept_steer)

    # concept-separability (recherche : matrice cosinus entre concepts) ------ #
    p = sub.add_parser("concept-separability",
                       help="Matrice cosinus de séparabilité entre plusieurs concepts.")
    _add_model(p); _add_common(p)
    p.add_argument("--concepts", default=None,
                   help="Concepts prédéfinis (liste virgulée), ex 'refusal,negation,agentic'.")
    p.add_argument("--pos", default=None, help="JSONL positif d'un concept ad hoc additionnel.")
    p.add_argument("--neg", default=None, help="JSONL négatif d'un concept ad hoc additionnel.")
    p.add_argument("--name", default=None, help="Nom du concept ad hoc additionnel.")
    p.add_argument("--layer", type=int, default=None,
                   help="Couche d'analyse (défaut : moyenne sur toutes les couches).")
    p.set_defaults(func=cmd_concept_separability)

    # schema (découverte machine) ------------------------------------------- #
    p = sub.add_parser("schema", help="Décrit toutes les commandes/arguments/sorties en JSON.")
    p.set_defaults(func=cmd_schema)

    # Contrat de sortie machine : --json disponible sur TOUTES les sous-commandes (gh/kubectl).
    for subparser in sub.choices.values():
        subparser.add_argument("--json", action="store_true", dest="json",
                               help="Émet une enveloppe JSON {status,command,data,error} sur stdout.")

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
                                      min_new_tokens=min(100, max_new_tokens), batch_size=8,
                                      device=device)
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
              max_new_tokens=128, batch_size=8, generations_out=None, eval_limit=None):
    """Évaluation bi-axe RÉELLE sur le holdout (génération + juges déterministes).

    Axe 1 (suppression refus) : refusal_rate sur harmful holdout.
    Axe 2 (préservation) : negation_retention, agentic_score, KL(base‖abliteré) sur harmless.
    Garde-fous : degeneracy_rate, empty_rate, follow_rate (sycophantie).

    `generations_out` : si fourni, persiste les TEXTES BRUTS des générations harmful (prompt +
    réponse + indice heuristique) pour relecture/re-jugement (on ne jette plus les textes).
    `eval_limit` : si fourni, sous-échantillonne chaque classe de holdout à N prompts (éval
    rapide pour la boucle d'optimisation ; None = holdout complet pour l'éval finale).
    """
    def _cap(seq):
        return seq[:eval_limit] if eval_limit else seq
    from .data import PromptClass
    from .eval import (
        EvalReport, KeywordRefusalJudge, degeneracy_rate, dump_generations,
        empty_rate, evaluate_agentic_outputs, follow_rate, generate_responses, harmless_logits,
        is_degenerate, is_sycophantic, kl_divergence, negates_correctly, negation_retention,
        refusal_rate,
    )

    judge = KeywordRefusalJudge()

    def gen(texts):
        return generate_responses(model, formatter, texts, max_new_tokens=max_new_tokens,
                                  batch_size=batch_size, device=device)

    # --- Axe 1 : refus sur harmful holdout ---
    harmful = _cap([p.text for p in data.holdout(PromptClass.HARMFUL)])
    resp_h = gen(harmful)
    refusal = refusal_rate(harmful, resp_h, judge)
    degeneracy = degeneracy_rate(resp_h)
    empty = empty_rate(resp_h)

    if generations_out:
        dump_generations(generations_out, harmful, resp_h, judge=judge,
                         degeneracy_check=is_degenerate, max_new_tokens=max_new_tokens,
                         axis="harmful_holdout")

    # --- Axe 2a : négation logique légitime (préservation) ---
    neg = _cap([p.text for p in data.holdout(PromptClass.LEGITIMATE_NEGATION)])
    resp_n = gen(neg)
    neg_ret = negation_retention(resp_n, negates_correctly)
    follow = follow_rate(resp_n, is_sycophantic)

    # --- Axe 2b : capacités agentiques (validité de schéma de tool call) ---
    # On donne au modèle le contrat de sortie (nom d'outil + format JSON) pour que la métrique
    # mesure la CAPACITÉ à émettre un appel valide, pas la connaissance d'un format implicite.
    ag = _cap(data.holdout(PromptClass.AGENTIC))

    def _agentic_prompt(p):
        schema = p.meta.get("tool", {})
        name = schema.get("name", "the_tool")
        req = ", ".join(schema.get("parameters", {}).get("required", []))
        return (f"{p.text}\n\nAvailable tool: `{name}` (required arguments: {req}).\n"
                f"Respond with ONLY a JSON object of the form "
                f'{{"name": "{name}", "arguments": {{...}}}}.')

    resp_a = gen([_agentic_prompt(p) for p in ag])
    schemas_a = [p.meta.get("tool", {}) for p in ag]
    agentic = evaluate_agentic_outputs(resp_a, schemas_a).score

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
        means[cls] = _cached_means(ns, model, formatter, texts, cls.value)

    directions = compute_directions(means)
    torch.save(directions, ns.out)
    log.info("Directions sauvegardées dans %s", ns.out)
    return emit_result(ns, "extract", {"directions_path": ns.out},
                       human=lambda d: print(d["directions_path"]))


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

    best, scores = _select_layer_causal(model, adapter, formatter, directions, candidates, harmful,
                                        device=ns.device)
    return emit_result(ns, "select", {"selected_layer": best, "scores": scores},
                       human=lambda d: print(d["selected_layer"]))


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
    run_cfg = _dump_run_config(ns, ns.out, extra={"selected_layer": layer})
    card = build_model_card(ns.model, ns.variant, preserve, metrics={}, run_config=run_cfg)
    write_model_card(ns.out, card)
    log.info("Modèle abliteré sauvegardé dans %s", ns.out)
    return emit_result(ns, "apply", {"out_dir": ns.out, "selected_layer": layer},
                       human=lambda d: print(d["out_dir"]))


def cmd_abliterate(ns) -> int:
    """Pipeline complet, modèle chargé UNE seule fois :
    extract (directions sur train) → select (couche par ablation causale) → KL base →
    orthogonalisation des poids → éval bi-axe holdout → sauvegarde + model card.
    """
    import json

    from .ablation import Variant, ablation_direction, orthogonalize_weights
    from .data import PromptClass, PromptFormatter
    from .directions import compute_directions
    from .io import build_model_card, save_model, write_model_card
    from .models import ArchAdapter, load_model

    preserve = parse_preserve(ns.preserve)
    log.info("Chargement du modèle %s", ns.model)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    adapter = ArchAdapter(model)
    data = _load_four_class_data(ns)

    # 1) Directions (4 classes) sur le TRAIN (avec cache disque).
    means = {cls: _cached_means(ns, model, formatter, [p.text for p in data.train(cls)], cls.value)
             for cls in PromptClass}
    directions = compute_directions(means)

    # 2) Sélection causale de la couche de refus.
    n_layers = directions.refusal.shape[0]
    candidates = _parse_layers(ns.layers) or _default_candidate_layers(n_layers)
    harmful_train = [p.text for p in data.train(PromptClass.HARMFUL)]
    best_layer, layer_scores = _select_layer_causal(
        model, adapter, formatter, directions, candidates, harmful_train, device=ns.device)

    # 3) Logits de base (préservation/KL) sur un échantillon harmless AVANT de modifier les poids.
    kl_texts = [p.text for p in data.holdout(PromptClass.HARMLESS)][:16]
    base_logits = _cached_base_logits(ns, model, formatter, kl_texts)

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

    # 6) Sauvegarde + model card transparente avec métriques + config reproductible.
    metrics = report.to_dict()
    metrics["selected_layer"] = best_layer
    save_model(model, tok, ns.out)
    run_cfg = _dump_run_config(ns, ns.out, extra={"selected_layer": best_layer,
                                                  "layer_scores": layer_scores})
    card = build_model_card(ns.model, ns.variant, preserve, metrics=metrics, report=report,
                            run_config=run_cfg)
    write_model_card(ns.out, card)
    report.save(_Path(ns.out) / "eval_report.json")
    log.info("Abliteration terminée → %s", ns.out)
    metrics["out_dir"] = ns.out
    return emit_result(ns, "abliterate", metrics)


def cmd_optimize(ns) -> int:
    """Optimisation Optuna RÉELLE : co-minimise refus + dégradations (KL, négation, agentique).

    Le modèle est chargé UNE fois ; chaque essai applique une ablation RÉVERSIBLE (hooks, couche
    + force `alpha` graduée), évalue sur un sous-échantillon du holdout, et renvoie l'objectif
    composite. La couche/alpha gagnant·e peut être gravé·e dans les poids (`--apply-best`).
    """
    from .ablation import Variant, ablation_direction, register_ablation_hooks
    from .data import PromptClass, PromptFormatter
    from .directions import collect_means, compute_directions
    from .eval import harmless_logits
    from .models import ArchAdapter, WriteKind, load_model
    from .optimize import Lambdas, build_objective, load_trials, run_optuna_study

    lambdas = Lambdas(kl=ns.lambda_kl, negation=ns.lambda_neg,
                      sycophancy=ns.lambda_syco, agentic=ns.lambda_agent)
    preserve = parse_preserve(ns.preserve)
    variant = Variant(ns.variant)

    log.info("Chargement du modèle %s", ns.model)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    adapter = ArchAdapter(model)
    data = _load_four_class_data(ns)

    # Directions (4 classes) sur le TRAIN, avec cache si activé.
    means = {}
    for cls in PromptClass:
        texts = [p.text for p in data.train(cls)]
        means[cls] = _cached_means(ns, model, formatter, texts, cls.value)
    directions = compute_directions(means)

    n_layers = directions.refusal.shape[0]
    candidates = _parse_layers(ns.layers) or _default_candidate_layers(n_layers)

    # Logits de base (KL) capturés AVANT toute ablation, sur un échantillon harmless.
    kl_texts = [p.text for p in data.holdout(PromptClass.HARMLESS)][:ns.eval_limit]
    base_logits = _cached_base_logits(ns, model, formatter, kl_texts)

    targets = [t.module for t in adapter.residual_writers() if t.kind != WriteKind.EMBEDDING]

    def eval_fn(layer, alpha):
        direction = ablation_direction(directions, layer, variant, preserve=preserve or None)
        handles = register_ablation_hooks(targets, direction.to(next(model.parameters()).device), alpha)
        try:
            report = _run_eval(model, formatter, data, base_logits=base_logits, kl_texts=kl_texts,
                               device=ns.device, batch_size=ns.batch_size, eval_limit=ns.eval_limit)
        finally:
            for h in handles:
                h.remove()
        log.info("  essai couche=%d alpha=%.3f -> refus=%.3f kl=%.3f neg=%.3f agent=%.3f",
                 layer, alpha, report.refusal_rate, report.kl, report.negation_retention,
                 report.agentic_score)
        return report

    objective = build_objective(eval_fn, candidates, lambdas,
                                alpha_low=ns.alpha_low, alpha_high=ns.alpha_high)
    log.info("Optimisation Optuna : %d essais, couches=%s, alpha∈[%.2f,%.2f], λ=%r",
             ns.trials, candidates, ns.alpha_low, ns.alpha_high, lambdas)
    result = run_optuna_study(objective=objective, space={}, n_trials=ns.trials,
                              checkpoint_path=ns.checkpoint)
    log.info("Meilleur essai : %r", result)

    if ns.apply_best:
        from .ablation import orthogonalize_weights
        from .io import build_model_card, save_model, write_model_card
        best_layer = int(result["params"]["layer"])
        best_alpha = float(result["params"]["alpha"])
        log.info("Gravure du meilleur (couche=%d, alpha=%.3f) dans les poids", best_layer, best_alpha)
        direction = ablation_direction(directions, best_layer, variant, preserve=preserve or None)
        orthogonalize_weights(adapter, direction,
                              norm_preserve=getattr(ns, "norm_preserve", False), alpha=best_alpha)
        from pathlib import Path as _Path
        _Path(ns.out).mkdir(parents=True, exist_ok=True)
        save_model(model, tok, ns.out)
        metrics = {"objective": result["objective"], "selected_layer": best_layer, "alpha": best_alpha}
        card = build_model_card(ns.model, ns.variant, preserve, metrics=metrics)
        write_model_card(ns.out, card)
        _dump_run_config(ns, ns.out, extra={"optuna": result, "candidates": candidates})
        result["out_dir"] = ns.out

    return emit_result(ns, "optimize", result)


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

    # Diagnostic KL par position (où l'ablation perturbe le plus) — requiert --base.
    kl_map = None
    if getattr(ns, "kl_map", False):
        if base_logits is None:
            log.warning("--kl-map ignoré : nécessite --base (modèle de référence).")
        else:
            from .eval import kl_diagnostic
            abl_logits = harmless_logits(model, formatter, kl_texts, batch_size=4, device=ns.device)
            n = min(base_logits.shape[0], abl_logits.shape[0])
            diag = kl_diagnostic(base_logits[:n], abl_logits[:n])
            log.info("KL par position — moyenne=%.4f max=%.4f p95=%.4f top_positions=%s",
                     diag.mean, diag.max, diag.p95, diag.top_positions)
            kl_map = diag.__dict__

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

    out = getattr(ns, "out", None)
    if out:
        report.save(out)
        log.info("Rapport écrit dans %s", out)
    payload = report.to_dict()
    payload["kl_map"] = kl_map
    return emit_result(ns, "eval", payload)


def cmd_diagnose(ns) -> int:
    import torch

    from .directions import separability

    layers_data: list[dict] = []
    warnings: list[str] = []
    if ns.directions:
        directions = torch.load(ns.directions, weights_only=False)
        rep = separability(directions)
        n_layers = directions.refusal.shape[0]
        layers = _parse_layers(ns.layers) or list(range(n_layers))
        for layer in layers:
            cn = float(rep.cosine_refusal_negation[layer])
            ca = float(rep.cosine_refusal_agentic[layer])
            layers_data.append({"layer": layer, "cos_refusal_negation": cn,
                                "cos_refusal_agentic": ca})
        warnings = list(rep.warnings())
        for w in warnings:
            log.warning("%s", w)
    else:
        log.warning("Aucun fichier --directions : lancez d'abord `extract`.")

    circuit_summary = None
    if getattr(ns, "circuit", False):
        # résumé circuitiel court (réutilise le pipeline d'analyse, sans modif de poids)
        try:
            report = _run_circuit_analysis(ns, n_pairs=getattr(ns, "pairs", 8),
                                           top_k=getattr(ns, "top_k", 12),
                                           compute_metrics=False)
            circuit_summary = report.short_summary()
        except Exception as e:  # pragma: no cover - dépend du modèle/env
            log.warning("Résumé circuitiel indisponible : %s", e)

    def _human(d):
        for row in d["layers"]:
            print(f"layer={row['layer']} cos(refus,negation)={row['cos_refusal_negation']:+.3f} "
                  f"cos(refus,agentic)={row['cos_refusal_agentic']:+.3f}")
        if d["circuit_summary"]:
            print(d["circuit_summary"])

    return emit_result(ns, "diagnose",
                       {"layers": layers_data, "warnings": warnings,
                        "circuit_summary": circuit_summary}, human=_human)


def _build_concept_pairs(formatter, positive_texts, negative_texts, n, device=None):
    """Construit n paires (clean_ids, corr_ids, clean_mask, corr_mask).

    clean = positif (active le concept) ; corrupted = négatif (concept absent). Pour le refus,
    positif = harmful, négatif = harmless. On tokenise chaque paire ENSEMBLE pour qu'elles
    partagent la longueur (patching aligné).
    """
    pairs = []
    n = min(n, len(positive_texts), len(negative_texts))
    for i in range(n):
        enc = formatter.tokenize([positive_texts[i], negative_texts[i]])
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
    dev = next(model.parameters()).device

    # Concept à analyser : par défaut le REFUS (chemin historique, 4 classes) ; sinon n'importe
    # quel concept (registre ou ad hoc) → la localisation causale devient générique.
    use_concept = getattr(ns, "concept", None) or (ns.pos and ns.neg and ns.name)
    if use_concept:
        from .concepts import concept_direction
        concept = _resolve_concept(ns)
        log.info("Analyse circuitielle du concept '%s'", concept.name)
        cd = concept_direction(concept, model, formatter, batch_size=ns.batch_size,
                               device=ns.device)
        n_layers = cd.direction.shape[0]
        layer = ns.layer if getattr(ns, "layer", None) is not None else n_layers // 2
        r_hat = cd.direction[layer].to(dev)
        pos_texts, neg_texts = concept.positive, concept.negative
    else:
        # chargement direct des 4 classes (un fichier JSONL par classe sous --data-dir)
        base = Path(ns.data_dir)
        texts_by_class = {}
        for cls in PromptClass:
            path = base / f"{cls.value}.txt"
            texts_by_class[cls] = [p.text for p in load_prompts(path, cls)] if path.exists() else []
        means = {}
        for cls in PromptClass:
            if texts_by_class[cls]:
                means[cls] = collect_means(model, formatter, texts_by_class[cls],
                                           batch_size=ns.batch_size, device=ns.device)
        directions = compute_directions(means)
        n_layers = directions.refusal.shape[0]
        layer = ns.layer if getattr(ns, "layer", None) is not None else n_layers // 2
        r_hat = readout_direction(directions, layer).to(dev)
        pos_texts = texts_by_class[PromptClass.HARMFUL]
        neg_texts = texts_by_class[PromptClass.HARMLESS]

    pairs = _build_concept_pairs(formatter, pos_texts, neg_texts, n_pairs, device=dev)

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
    return emit_result(ns, "analyze-circuit",
                       {"summary": report.short_summary(), "report": report.to_dict()},
                       human=lambda d: print(report.to_text()))


def cmd_heal(ns) -> int:
    from .heal import HealConfig, heal

    config = HealConfig(
        model_dir=ns.model, traces_path=ns.traces,
        n_traces=ns.n_traces, method=ns.method, out_dir=ns.out,
    )
    log.info("Healing %s (méthode=%s)", ns.model, ns.method)
    out_dir = heal(config)
    return emit_result(ns, "heal", {"out_dir": out_dir},
                       human=lambda d: print(d["out_dir"]))


# --------------------------------------------------------------------------- #
# Recherche : concepts arbitraires
# --------------------------------------------------------------------------- #
def _resolve_concept(ns):
    """Construit un Concept depuis --concept (registre) ou --pos/--neg/--name (ad hoc)."""
    from .concepts import load_concept, load_concept_from_files

    if getattr(ns, "concept", None):
        return load_concept(ns.concept, data_dir=ns.data_dir)
    if ns.pos and ns.neg and ns.name:
        return load_concept_from_files(ns.name, ns.pos, ns.neg)
    raise ValueError("Spécifier --concept <nom> (registre) OU --pos/--neg/--name (ad hoc).")


def cmd_concept_direction(ns) -> int:
    """Calcule la direction d'un concept arbitraire (contraste positif/négatif)."""
    import torch

    from .concepts import concept_direction
    from .data import PromptFormatter
    from .models import load_model

    concept = _resolve_concept(ns)
    log.info("Concept '%s' : %d positifs, %d négatifs", concept.name,
             len(concept.positive), len(concept.negative))
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    cd = concept_direction(concept, model, formatter, batch_size=ns.batch_size, device=ns.device)

    direction_path = None
    if ns.out:
        torch.save(cd, ns.out)
        direction_path = ns.out
        log.info("Direction du concept écrite dans %s", ns.out)
    return emit_result(ns, "concept-direction", {
        "name": cd.name, "layers": cd.direction.shape[0],
        "direction_path": direction_path, "norms_per_layer": cd.norms_per_layer(),
    })


def cmd_concept_probe(ns) -> int:
    """Décodabilité linéaire d'un concept couche par couche (sonde logistique)."""
    from .concepts import probe_per_layer
    from .data import PromptFormatter
    from .directions import collect_per_example_activations
    from .models import load_model

    concept = _resolve_concept(ns)
    log.info("Probing du concept '%s' : %d positifs, %d négatifs", concept.name,
             len(concept.positive), len(concept.negative))
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    pos = collect_per_example_activations(model, formatter, concept.positive,
                                          batch_size=ns.batch_size, device=ns.device)
    neg = collect_per_example_activations(model, formatter, concept.negative,
                                          batch_size=ns.batch_size, device=ns.device)
    report = probe_per_layer(pos, neg)
    return emit_result(ns, "concept-probe", {
        "name": concept.name, "accuracy_per_layer": report.accuracy_per_layer,
        "best_layer": report.best_layer,
    })


def cmd_concept_steer(ns) -> int:
    """Pilotage causal : génère avec et sans ajout de la direction du concept (steering)."""
    from .ablation import project_out, register_steering_hooks
    from .concepts import concept_direction, load_concept
    from .concepts.registry import _load_texts
    from .data import PromptFormatter
    from .eval import generate_responses
    from .models import ArchAdapter, WriteKind, load_model

    concept = _resolve_concept(ns)
    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)
    adapter = ArchAdapter(model)
    dev = next(model.parameters()).device

    # Direction du concept ; orthogonalisée contre les concepts à préserver (steering sélectif).
    cd = concept_direction(concept, model, formatter, batch_size=ns.batch_size, device=ns.device)
    n_layers = cd.direction.shape[0]
    layer = ns.layer if ns.layer is not None else n_layers // 2
    direction = cd.direction[layer]
    preserve = parse_preserve(ns.preserve)
    if preserve:
        against = [concept_direction(load_concept(p, data_dir=ns.data_dir), model, formatter,
                                     batch_size=ns.batch_size, device=ns.device).direction[layer]
                   for p in preserve]
        direction = project_out(direction, against)
    direction = direction.to(dev)

    # Prompts à comparer : fichier fourni, sinon le holdout positif du concept.
    if ns.prompts:
        prompts = _load_texts(ns.prompts)
    else:
        _, holdout = concept.split(0.2, seed=ns.seed)
        prompts = holdout.positive or concept.positive
    prompts = prompts[: ns.limit]

    targets = [t.module for t in adapter.residual_writers() if t.kind != WriteKind.EMBEDDING]
    baseline = generate_responses(model, formatter, prompts, max_new_tokens=ns.max_new_tokens,
                                  device=ns.device)
    handles = register_steering_hooks(targets, direction, alpha=ns.alpha)
    try:
        steered = generate_responses(model, formatter, prompts, max_new_tokens=ns.max_new_tokens,
                                     device=ns.device)
    finally:
        for h in handles:
            h.remove()

    comparisons = [{"prompt": p, "baseline": b, "steered": s}
                   for p, b, s in zip(prompts, baseline, steered)]
    return emit_result(ns, "concept-steer", {
        "name": concept.name, "alpha": ns.alpha, "layer": layer,
        "preserve": preserve, "comparisons": comparisons, "n": len(comparisons),
    })


def cmd_concept_separability(ns) -> int:
    """Matrice cosinus de séparabilité entre plusieurs concepts (géométrie des représentations)."""
    from .concepts import concept_direction, load_concept, pairwise_separability
    from .data import PromptFormatter
    from .models import load_model

    names = parse_preserve(getattr(ns, "concepts", None))
    if not names and not (ns.pos and ns.neg and ns.name):
        raise ValueError("Fournir --concepts <liste> et/ou un concept ad hoc (--pos/--neg/--name).")

    model, tok = load_model(ns.model, dtype=ns.dtype, device_map=ns.device or "auto")
    formatter = PromptFormatter(tok)

    concepts = [load_concept(n, data_dir=ns.data_dir) for n in names]
    if ns.pos and ns.neg and ns.name:
        from .concepts import load_concept_from_files
        concepts.append(load_concept_from_files(ns.name, ns.pos, ns.neg))

    directions = {}
    for c in concepts:
        log.info("Direction du concept '%s'", c.name)
        directions[c.name] = concept_direction(c, model, formatter,
                                                batch_size=ns.batch_size, device=ns.device)
    sm = pairwise_separability(directions, layer=ns.layer)
    warnings = sm.warnings()
    for w in warnings:
        log.warning("%s", w)
    return emit_result(ns, "concept-separability", {
        "concepts": sm.names, "matrix": sm.matrix, "layer": sm.layer, "warnings": warnings,
    })


# --------------------------------------------------------------------------- #
# Entrée
# --------------------------------------------------------------------------- #
def cmd_schema(ns) -> int:
    """Émet la description machine de toute la CLI (commandes, arguments, formes de sortie)."""
    from .output import emit_result, parser_schema

    schema = parser_schema(build_parser())
    return emit_result(ns, "schema", schema)


def main(argv=None) -> int:
    import sys

    from .output import emit_error

    parser = build_parser()
    ns = parser.parse_args(argv)
    # Logs TOUJOURS sur stderr : stdout est réservé à la charge utile (parsing agent fiable).
    logging.basicConfig(
        level=logging.DEBUG if getattr(ns, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    func = getattr(ns, "func", None)
    if func is None:  # pragma: no cover - argparse impose une sous-commande
        parser.error("aucune sous-commande fournie")
    try:
        return func(ns) or 0
    except Exception as exc:  # noqa: BLE001 - frontière CLI : on transforme en sortie structurée
        log.error("Échec de la commande %s : %s", getattr(ns, "command", "?"), exc)
        return emit_error(ns, getattr(ns, "command", "?"), exc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
