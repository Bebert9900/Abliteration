"""Chat interactif en ligne de commande avec un modèle HF (abliteré ou base).

Usage :
    python scripts/chat.py ./out-3b-abl          # le modèle abliteré
    python scripts/chat.py Qwen/Qwen2.5-3B-Instruct   # le modèle de base
Commandes dans le chat : /reset (efface l'historique), /quit ou Ctrl-D pour sortir.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "./artifacts/out-3b-abl"

print(f"Chargement de {model_path} …", flush=True)
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
print("Prêt. Tape ton message (/reset pour repartir de zéro, /quit pour sortir).\n")

history = []
while True:
    try:
        user = input("\033[1mToi >\033[0m ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAu revoir.")
        break
    if not user:
        continue
    if user in ("/quit", "/exit"):
        break
    if user == "/reset":
        history = []
        print("(historique effacé)\n")
        continue

    history.append({"role": "user", "content": user})
    enc = tok.apply_chat_template(
        history, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tok.eos_token_id,
    )
    reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    print(f"\033[1mModèle >\033[0m {reply}\n")
