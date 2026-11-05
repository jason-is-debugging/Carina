"""SFT corpus: download + process + Dataset with prompt masking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

_ALIAS = {"tinystories": "roneneldan/TinyStories"}


class SftTokenizer(Protocol):
    """Minimal tokenizer surface needed by SftDataset."""

    def encode(self, text: str) -> list[int]: ...


def _extract_pair(obj: dict[str, object]) -> tuple[str, str]:
    """Normalize raw row into (prompt, response) strings."""
    prompt = obj.get("prompt", obj.get("instruction", ""))
    response = obj.get("response", obj.get("output", obj.get("text", "")))
    prompt_s: str = prompt.strip() if isinstance(prompt, str) else ""
    response_s: str = response.strip() if isinstance(response, str) else ""
    return prompt_s, response_s


def download_finetune_dataset(
    source: str,
    save_path: Path,
    max_samples: int = 2000,
) -> Path:
    """Download finetune raw corpus to JSONL with prompt/response rows."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    hf_name: str = _ALIAS.get(source, source)
    split: str = f"train[:{max_samples}]"
    ds = load_dataset(hf_name, split=split)
    count: int = 0
    with open(save_path, "w", encoding="utf-8") as fh:
        for row in ds:
            prompt, response = _extract_pair(row)
            if len(response) == 0:
                continue
            fh.write(
                json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False)
                + "\n"
            )
            count += 1
    return save_path


def process_finetune_dataset(
    raw_path: Path,
    save_path: Path,
    min_chars: int = 8,
    max_samples: int = -1,
) -> Path:
    """Clean raw JSONL into {"prompt": ..., "response": ...} rows."""
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
            prompt, response = _extract_pair(json.loads(line))
            if len(response) < min_chars:
                continue
            fout.write(
                json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False)
                + "\n"
            )
            kept += 1
    assert kept > 0, f"[datasets.sft.process] kept 0 rows from {str(raw_path)!r}"
    return save_path


class SftDataset(Dataset):
    """SFT dataset: item -> (input_ids, labels) with prompt masked.

    Causal shift: inputs[t] predicts labels[t]=next token.
    Labels for prompt-internal positions and pads are ``ignore_index``.

    pack=False: one pair per sample, padded when short (old behaviour, which
        wasted a lot of padding on short samples).
    pack=True: prompt+response+eos are concatenated into one long stream and
        cut into blocks of width, so several samples share a block while
        prompt positions stay masked (including across block boundaries).
    """

    pairs: list[tuple[list[int], list[int]]]
    packed: list[tuple[list[int], list[bool]]] | None
    seq_len: int
    pad_id: int
    ignore_index: int

    def __init__(
        self,
        processed_path: Path,
        seq_len: int,
        tokenizer: SftTokenizer,
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
        self.pairs = []
        with open(processed_path, encoding="utf-8") as fh:
            for line in fh:
                if max_samples > 0 and len(self.pairs) >= max_samples:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                prompt, response = _extract_pair(json.loads(line))
                if len(response) == 0:
                    continue
                prompt_ids: list[int] = list(tokenizer.encode(prompt)) if prompt else []
                response_ids: list[int] = list(tokenizer.encode(response))

                self.pairs.append((prompt_ids, response_ids))
        self.packed = None
        if pack:
            # A flat stream plus an is_prompt flag, cut into non-overlapping
            # blocks of width=seq_len+1.
            # Prompt tokens are all flagged True (including the first token,
            # never learned); response and eos tokens are flagged False (to
            # be learned, including eos across sample boundaries).
            width: int = seq_len + 1
            flat: list[int] = []
            is_prompt: list[bool] = []
            for prompt_ids, response_ids in self.pairs:
                flat.extend(prompt_ids)
                is_prompt.extend([True] * len(prompt_ids))
                flat.extend(response_ids)
                is_prompt.extend([False] * len(response_ids))
                flat.append(eos_id)
                is_prompt.append(False)
            self.packed = []
            for start in range(0, len(flat), width):
                ids_chunk: list[int] = flat[start:start + width]
                mask_chunk: list[bool] = is_prompt[start:start + width]
                # The trailing block is stored unpadded; __getitem__ pads and
                # masks it.
                self.packed.append((ids_chunk, mask_chunk))

    def __len__(self) -> int:
        if self.packed is not None:
            return len(self.packed)
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.packed is not None:
            ids_chunk, mask_chunk = self.packed[idx]
            width: int = self.seq_len + 1
            real: int = len(ids_chunk)
            full: list[int] = (
                ids_chunk + [self.pad_id] * (width - real)
            )
            full_mask: list[bool] = mask_chunk + [True] * (width - real)
            inputs: list[int] = full[:-1]
            raw_labels: list[int] = full[1:]
            labels: list[int] = []
            for t, lab in enumerate(raw_labels):
                # mask[t] is True -> the target is inside the prompt or a
                # pad, so it is not learned.
                # Note: mask[t] marks the ownership of inputs[t+1], which
                # lines up exactly with labels[t].
                if full_mask[t + 1] or t + 1 >= real:
                    labels.append(self.ignore_index)
                else:
                    labels.append(lab)
            return torch.tensor(inputs, dtype=torch.long), torch.tensor(
                labels, dtype=torch.long
            )
        prompt_ids, response_ids = self.pairs[idx]
        width: int = self.seq_len + 1
        full: list[int] = (prompt_ids + response_ids)[:width]
        real: int = len(full)
        full = full + [self.pad_id] * (width - real)
        inputs: list[int] = full[:-1]
        raw_labels: list[int] = full[1:]
        prompt_eff: int = min(len(prompt_ids), real)
        labels: list[int] = []
        for t, lab in enumerate(raw_labels):
            if t + 1 < prompt_eff or t + 1 >= real:
                labels.append(self.ignore_index)
            else:
                labels.append(lab)
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )
