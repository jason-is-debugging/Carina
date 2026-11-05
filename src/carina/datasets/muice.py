"""Muice corpus: download + persona rename + process.

Real format: {"system": "你是一个名为沐雪的可爱AI女孩子",
"conversation": [{"human": ..., "assistant": ...}, ...]} (3637 train rows).
Rename logic: the original character names (沐雪/沐沐/沐妮卡/慕雪/Muice/...)
are all replaced with our persona (CJK -> 柳如烟, ASCII -> Carina), including
the system field, so the model never learns to call itself 沐雪.

No config import; names are passed in as arguments.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def rename_persona(
    text: str,
    orig_names: tuple[str, ...],
    name_zh: str,
    name_en: str,
) -> str:
    """Replace source role names with our persona names.

    Args:
        text: Raw utterance.
        orig_names: Source name variants (mixed zh/en).
        name_zh: Replacement for CJK variants (e.g. 柳如烟).
        name_en: Replacement for ASCII variants (e.g. Carina).

    Returns:
        Renamed text. Longest names replaced first.
    """
    out: str = text
    for name in sorted(orig_names, key=len, reverse=True):
        if not name:
            continue
        target = name_zh if any(ord(c) > 127 for c in name) else name_en
        out = out.replace(name, target)
    return out


def _rename_obj(
    obj: object,
    orig_names: tuple[str, ...],
    name_zh: str,
    name_en: str,
) -> object:
    """Recursively rename all strings inside a JSON structure."""
    if isinstance(obj, str):
        return rename_persona(obj, orig_names, name_zh, name_en)
    if isinstance(obj, list):
        return [_rename_obj(v, orig_names, name_zh, name_en) for v in obj]
    if isinstance(obj, dict):
        return {
            k: _rename_obj(v, orig_names, name_zh, name_en)
            for k, v in obj.items()
        }
    return obj


def download_muice_dataset(
    hf_id: str,
    save_path: Path,
    max_samples: int = -1,
) -> Path:
    """Download Muice train split to raw JSONL (one row per sample)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    split: str = "train" if max_samples <= 0 else f"train[:{max_samples}]"
    ds = load_dataset(hf_id, split=split)
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in ds)
    return save_path


def process_muice_dataset(
    raw_path: Path,
    save_path: Path,
    orig_names: tuple[str, ...],
    name_zh: str,
    name_en: str,
    min_chars: int = 8,
    max_samples: int = -1,
) -> Path:
    """Rename persona + clean raw JSONL into {"prompt","response"} rows.

    ShareGPT row -> the first human turn becomes the prompt and the remaining
    gpt replies are joined into the response.
    Accepts {"prompt","response"} / {"conversations":[...]} / {"text":...}.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    kept: int = 0
    with (
        open(raw_path, encoding="utf-8") as fin,
        open(save_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if max_samples > 0 and kept >= max_samples:
                break
            line = line.strip()
            if len(line) == 0:
                continue
            row = _rename_obj(json.loads(line), orig_names, name_zh, name_en)
            assert isinstance(row, dict)
            prompt, response = _to_pair(row)
            if len(response) < min_chars:
                continue
            fout.write(
                json.dumps({"prompt": prompt, "response": response},
                           ensure_ascii=False) + "\n"
            )
            kept += 1
    return save_path


def _to_pair(row: dict) -> tuple[str, str]:
    """Normalize one renamed row into (prompt, response)."""
    convs = row.get("conversation", row.get("conversations"))
    if isinstance(convs, list) and convs:
        humans: list[str] = []
        bots: list[str] = []
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            # Real Muice format: {"human": ..., "assistant": ...}
            if "human" in turn or "assistant" in turn:
                h = str(turn.get("human", "")).strip()
                a = str(turn.get("assistant", "")).strip()
                if h:
                    humans.append(h)
                if a:
                    bots.append(a)
                continue
            # ShareGPT compatible: {"from": ..., "value": ...}
            role = str(turn.get("from", turn.get("role", ""))).lower()
            val = str(turn.get("value", turn.get("content", ""))).strip()
            if not val:
                continue
            if role in ("human", "user"):
                humans.append(val)
            elif role in ("gpt", "assistant", "bot"):
                bots.append(val)
        prompt = humans[0] if humans else ""
        response = "\n".join(bots if bots else humans[1:])
        return prompt.strip(), response.strip()
    prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
    response = str(
        row.get("response", row.get("output", row.get("text", "")))).strip()
    return prompt, response
