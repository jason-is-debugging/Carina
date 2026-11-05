"""LCCC chit-chat corpus: download + process.

Source: silver/lccc (MIT). The load_dataset script is deprecated (lccc.py
no longer supports it); the raw file lccc_base_train.jsonl.gz is pulled
directly with hf_hub_download.
Line format: a JSON list, one utterance per turn, already segmented with
spaces between Chinese characters, e.g.
["你 去 那儿 竟然 不喊 我", "道歉 ！ ！ 再有 时间 找 你 去"].

No config import; all values are passed in as arguments.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

LCCC_TRAIN_FILE = "lccc_base_train.jsonl.gz"


def _despace(text: str) -> str:
    """Strip the segmentation spaces: '你 去 那儿' -> '你去那儿'."""
    return text.replace(" ", "").replace("\u3000", "").strip()


def _parse_session(line: str) -> list[str]:
    """One line -> list of de-spaced utterances, empty list on parse failure."""
    try:
        turns = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(turns, list):
        return []
    out: list[str] = []
    for t in turns:
        if not isinstance(t, str):
            continue
        t = _despace(t)
        if t:
            out.append(t)
    return out


def download_lccc_dataset(
    hf_id: str,
    save_path: Path,
    filename: str = LCCC_TRAIN_FILE,
) -> Path:
    """Download the LCCC train .gz to save_path (decompressed to jsonl,
    one session per line)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path = Path(
        hf_hub_download(hf_id, filename, repo_type="dataset")
    )
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(
        save_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if line:
                fout.write(line + "\n")
    return save_path


def process_lccc_pretrain(
    raw_path: Path,
    save_path: Path,
    min_chars: int = 8,
    max_chars: int = 512,
    max_samples: int = 70000,
) -> Path:
    """raw session -> pretrain {"text": turns joined by newlines}."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    kept: int = 0
    with open(raw_path, encoding="utf-8") as fin, open(
        save_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if kept >= max_samples:
                break
            turns = _parse_session(line)
            if len(turns) < 2:
                continue
            text = "\n".join(turns)
            if not (min_chars <= len(text) <= max_chars):
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
    assert kept > 0, f"[datasets.lccc] kept 0 rows from {str(raw_path)!r}"
    return save_path


def process_lccc_sft(
    raw_path: Path,
    save_path: Path,
    min_chars: int = 8,
    max_samples: int = 5000,
    skip: int = 0,
) -> Path:
    """raw session -> sft {"prompt": earlier turns joined, "response": last turn}.

    skip: skip the first N usable sessions, so pretrain and sft draw from
    different segments.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    kept: int = 0
    seen: int = 0
    with open(raw_path, encoding="utf-8") as fin, open(
        save_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if kept >= max_samples:
                break
            turns = _parse_session(line)
            if len(turns) < 2 or len(turns[-1]) < min_chars:
                continue
            if seen < skip:
                seen += 1
                continue
            prompt = "\n".join(turns[:-1])
            response = turns[-1]
            fout.write(
                json.dumps({"prompt": prompt, "response": response},
                           ensure_ascii=False) + "\n"
            )
            kept += 1
    assert kept > 0, f"[datasets.lccc] kept 0 sft rows from {str(raw_path)!r}"
    return save_path


def copy_file(src: Path, dst: Path) -> Path:
    """Copy a small file directly (fallback)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst
