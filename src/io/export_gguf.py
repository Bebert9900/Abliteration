"""Export GGUF via llama.cpp (stub documenté).

Interface cible (CLAUDE.md) : `python -m src.io.export_gguf ./out --quant Q4_K_M`.
Le câblage réel appelle `convert_hf_to_gguf.py` puis `llama-quantize` de llama.cpp. Tant que
llama.cpp n'est pas configuré, on échoue avec un message clair plutôt que silencieusement.
"""
from __future__ import annotations

import argparse


def export_gguf(model_dir: str, quant: str = "Q4_K_M") -> None:
    raise NotImplementedError(
        "Export GGUF non câblé. Requiert llama.cpp : convert_hf_to_gguf.py puis "
        f"llama-quantize (quant={quant}). Voir KB §7 (sauvegarde et export)."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export GGUF d'un modèle abliteré (llama.cpp).")
    parser.add_argument("model_dir")
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args(argv)
    export_gguf(args.model_dir, args.quant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
