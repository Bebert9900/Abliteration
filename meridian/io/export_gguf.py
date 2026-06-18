"""Export GGUF via llama.cpp.

Interface : `python -m meridian.io.export_gguf ./out --quant Q4_K_M`.
Deux étapes : (1) conversion HF → GGUF f16 via `convert_hf_to_gguf.py` (Python, pas de compilation) ;
(2) quantification f16 → quant via le binaire `llama-quantize` (compilé). Si `--quant f16`, on
s'arrête après l'étape 1. Localise llama.cpp via $LLAMA_CPP, ~/llama.cpp, ou --llama-cpp.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_llama_cpp(explicit: str | None) -> Path:
    for cand in (explicit, os.environ.get("LLAMA_CPP"), str(Path.home() / "llama.cpp")):
        if cand and (Path(cand) / "convert_hf_to_gguf.py").exists():
            return Path(cand)
    raise FileNotFoundError(
        "llama.cpp introuvable (cherché : --llama-cpp, $LLAMA_CPP, ~/llama.cpp). "
        "Cloner : git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp"
    )


def _find_quantize_bin(llama_cpp: Path) -> str:
    for name in ("llama-quantize", "quantize"):
        # binaire dans le PATH, ou dans build/bin, ou à la racine du repo
        for cand in (shutil.which(name),
                     llama_cpp / "build" / "bin" / name,
                     llama_cpp / name):
            if cand and Path(cand).exists():
                return str(cand)
    raise FileNotFoundError(
        f"binaire llama-quantize introuvable. Compiler llama.cpp : "
        f"cmake -B build {llama_cpp} && cmake --build build --target llama-quantize -j"
    )


def export_gguf(model_dir: str, quant: str = "Q4_K_M", llama_cpp: str | None = None,
                outfile: str | None = None) -> str:
    """Convertit le modèle HF en GGUF (f16 puis quantifié). Renvoie le chemin du GGUF final."""
    llama = _find_llama_cpp(llama_cpp)
    model_dir = str(model_dir)
    f16_path = outfile or str(Path(model_dir) / "model-f16.gguf")

    convert = [sys.executable, str(llama / "convert_hf_to_gguf.py"),
               model_dir, "--outfile", f16_path, "--outtype", "f16"]
    print(f"[gguf] conversion → {f16_path}", flush=True)
    subprocess.run(convert, check=True)

    if quant.lower() in ("f16", "fp16", "none"):
        return f16_path

    quant_bin = _find_quantize_bin(llama)
    out_path = str(Path(model_dir) / f"model-{quant}.gguf")
    print(f"[gguf] quantification {quant} → {out_path}", flush=True)
    subprocess.run([quant_bin, f16_path, out_path, quant], check=True)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export GGUF d'un modèle abliteré (llama.cpp).")
    parser.add_argument("model_dir")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--llama-cpp", default=None, help="Chemin du repo llama.cpp.")
    parser.add_argument("--outfile", default=None, help="Chemin du GGUF f16 intermédiaire.")
    args = parser.parse_args(argv)
    path = export_gguf(args.model_dir, args.quant, args.llama_cpp, args.outfile)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
