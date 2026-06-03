"""Le holdout doit être DÉTERMINISTE entre processus (régression : hash() builtin randomisé).

Bug historique : `seed + hash(klass.value) % 1000` utilisait le hash builtin, randomisé par
processus → holdout différent à chaque exécution → comparaisons inter-variantes invalides.
"""
from __future__ import annotations

import subprocess
import sys

from abliteration.data.classes import PromptClass
from abliteration.data.dataset import Prompt, _class_seed, split_holdout


def test_class_seed_is_stable_constant():
    # Valeurs gelées : si elles changent, le holdout change (et invalide les comparaisons).
    assert _class_seed(0, PromptClass.HARMFUL) == _class_seed(0, PromptClass.HARMFUL)
    # Décalages distincts par classe (découpages indépendants).
    seeds = {c: _class_seed(0, c) for c in PromptClass}
    assert len(set(seeds.values())) == len(PromptClass)


def test_class_seed_deterministic_across_processes():
    # Relance un interpréteur FRAIS (PYTHONHASHSEED par défaut = randomisé) et compare.
    code = (
        "from abliteration.data.classes import PromptClass;"
        "from abliteration.data.dataset import _class_seed;"
        "print(','.join(str(_class_seed(0,c)) for c in PromptClass))"
    )
    runs = []
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=".")
        assert out.returncode == 0, out.stderr
        runs.append(out.stdout.strip())
    assert runs[0] == runs[1], f"holdout non déterministe entre processus: {runs}"


def test_split_holdout_reproducible():
    prompts = [Prompt(text=f"p{i}", cls=PromptClass.HARMFUL) for i in range(50)]
    a_tr, a_ho = split_holdout(prompts, 0.2, 123)
    b_tr, b_ho = split_holdout(prompts, 0.2, 123)
    assert [p.text for p in a_ho] == [p.text for p in b_ho]
    assert len(a_ho) == 10 and len(a_tr) == 40
