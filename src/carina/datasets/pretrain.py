"""Pretrain corpus: download + process + Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

_ALIAS = {"tinystories": "roneneldan/TinyStories"}


class TextTokenizer(Protocol):
    """Minimal tokenizer surface needed by PretrainDataset."""

    def encode(self, text: str) -> list[int]: ...


def _extract_text(obj: dict[str, object]) -> str:
    """Pick text from common raw fields, return stripped string."""
    for key in ("text", "story", "content", "document"):
        val = obj.get(key)
        if isinstance(val, str) and len(val.strip()) > 0:
            return val.strip()
    return ""


def download_pretrain_dataset(
    source: str,
    save_path: Path,
    max_samples: int = 2000,
) -> Path:
    """Download pretrain raw corpus to JSONL with {"text": ...} rows."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    hf_name: str = _ALIAS.get(source, source)
    split: str = f"train[:{max_samples}]"
    ds = load_dataset(hf_name, split=split)
    count: int = 0
    with open(save_path, "w", encoding="utf-8") as fh:
        for row in ds:
            text: str = _extract_text(row)
            if len(text) == 0:
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return save_path


def process_pretrain_dataset(
    raw_path: Path,
    save_path: Path,
    min_chars: int = 16,
    max_samples: int = -1,
) -> Path:
    """Clean raw JSONL into {"text": ...} rows for pretraining."""
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
            text: str = _extract_text(json.loads(line))
            if len(text) < min_chars:
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
    return save_path


class PretrainDataset(Dataset):
    """Causal-LM dataset: item -> (input_ids, labels) [seq_len].

    pack=False: one row per sample, padded when short (old behaviour, which
        wasted a lot of padding on short samples).
    pack=True: the whole corpus is concatenated into one long stream of
        [ids + eos] and cut into non-overlapping blocks of width=seq_len+1,
        so there is almost no padding (several short samples share a block).
    """

    samples: list[list[int]]
    seq_len: int
    pad_id: int
    ignore_index: int

    def __init__(
        self,
        processed_path: Path,
        seq_len: int,
        tokenizer: TextTokenizer,
        max_samples: int = -1,
        pad_id: int = 0,
        ignore_index: int = -100,
        pack: bool = False,
        eos_id: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len: int = seq_len
        self.pad_id: int = pad_id
        self.ignore_index: int = ignore_index
        self.samples = []
        raw: list[list[int]] = []
        with open(processed_path, encoding="utf-8") as fh:
            for line in fh:
                if max_samples > 0 and len(raw) >= max_samples:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                text: str = _extract_text(json.loads(line))
                if len(text) == 0:
                    continue
                ids: list[int] = list(tokenizer.encode(text))
                raw.append(ids)
        if pack:
            # The trailing block may be shorter than width; store it
            # unpadded and let __getitem__ pad and mask it, so padding is
            # never confused with a real eos (Qwen's pad and eos share an id).
            width: int = seq_len + 1
            flat: list[int] = []
            for ids in raw:
                flat.extend(ids)
                flat.append(eos_id)
            for start in range(0, len(flat), width):
                self.samples.append(flat[start:start + width])
        else:
            self.samples = raw

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        ids: list[int] = self.samples[idx]
        width: int = self.seq_len + 1
        real_len: int = min(len(ids), width)
        if len(ids) >= width:
            window: list[int] = ids[:width]
        else:
            window = ids + [self.pad_id] * (width - len(ids))
        inputs: Tensor = torch.tensor(window[:-1], dtype=torch.long)
        labels: Tensor = torch.tensor(window[1:], dtype=torch.long)
        if real_len <= self.seq_len:
            # Labels at j >= real_len - 1 predict pad -> ignore.
            # (real_len==0 edge: all ignored.)
            first_pad: int = max(real_len - 1, 0)
            labels[first_pad:] = self.ignore_index
        return inputs, labels
