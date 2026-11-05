"""SFT merge: weighted upsampling + shuffling -> finetune_merged.jsonl.

No config import; sources/weights/seed as arguments.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


def merge_sft(
    sources: list[Path],
    weights: list[float],
    out_path: Path,
    seed: int = 42,
) -> Path:
    """Weighted mix of processed SFT files into one merged JSONL.

    weight>=1: repeat the whole file floor(w) times + sample the remainder;
    weight<1: randomly subsample by that ratio.
    """
    assert len(sources) == len(weights) and sources, \
        "[datasets.merge] sources/weights mismatch or empty"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    mixed: list[str] = []
    for src, w in zip(sources, weights):
        lines = [ln for ln in open(src, encoding="utf-8")
                 if ln.strip()]
        if not lines:
            continue
        full, frac = int(math.floor(w)), w - math.floor(w)
        mixed.extend(lines * full)
        if frac > 1e-9:
            k = max(1, int(round(len(lines) * frac)))
            mixed.extend(rng.sample(lines, min(k, len(lines))))
    rng.shuffle(mixed)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.writelines(mixed)
    assert mixed, "[datasets.merge] merged 0 rows"
    return out_path
