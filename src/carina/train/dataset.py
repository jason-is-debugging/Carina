"""Dataset and DataLoader helpers for Carina SFT.

Implements a packed SFT dataset that produces ``(input_ids, labels)``
pairs where every non-assistant token is masked to ``-100``. We also
ship a tiny dummy dataset used by the smoke tests.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CarinaSFTDataset(Dataset):
    """SFT dataset that turns conversations into packed token tensors.

    Args:
        records: Iterable of dicts each shaped like
            ``{"conversations": [{"role": ..., "content": ...}, ...]}``.
        tokenizer: A :class:`carina.tokenizer.CarinaTokenizer`.
        max_length: Padded sequence length for every sample.
        system: Optional default system prompt inserted before every
            conversation that does not already have one.
    """

    def __init__(
        self,
        records: Sequence[dict],
        tokenizer,
        max_length: int = 1024,
        system: str | None = "You are Carina, a warm virtual companion.",
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        """Return ``(input_ids, labels)`` for the ``index``-th record."""
        record = self.records[index]
        if isinstance(record, dict) and "input_ids" in record and "labels" in record:
            return (
                torch.tensor(record["input_ids"], dtype=torch.long),
                torch.tensor(record["labels"], dtype=torch.long),
            )
        convo = record.get("conversations") if isinstance(record, dict) else None
        if not convo:
            raise ValueError(f"Record at {index} has no conversations field")
        messages: list[dict] = []
        if self.system and not any(m.get("role") == "system" for m in convo):
            messages.append({"role": "system", "content": self.system})
        for turn in convo:
            role = turn.get("role")
            if role not in ("system", "user", "assistant"):
                continue
            messages.append({"role": role, "content": turn.get("content", "")})
        text = self.tokenizer.apply_chat_template(messages, system=self.system)
        ids = self.tokenizer.encode(text)
        if len(ids) > self.max_length:
            # Keep the END (most recent turns = assistant reply) — discard old history.
            ids = ids[len(ids) - self.max_length :]
        if len(ids) < self.max_length:
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        labels = self.tokenizer.build_labels(ids)
        # Filter: skip records with fewer than 16 supervised tokens (broken by truncation).
        if sum(1 for l in labels if l != -100) < 16:
            return self.__getitem__((index + 1) % len(self))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DummySFTDataset(Dataset):
    """Tiny in-memory SFT dataset for smoke tests."""

    TEMPLATE = (
        "<|bos|><|system|>You are Carina.<|user|>How are you?<|eos|>"
        "<|assistant|>I'm fine, thank you.<|eos|>"
        "<|user|>Tell me a joke.<|eos|><|assistant|>Why did the chicken cross the road?<|eos|>"
    )

    def __init__(self, tokenizer, num_samples: int = 8, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        base_ids = tokenizer.encode(self.TEMPLATE)[: max_length - 2]
        base_ids = [tokenizer.bos_token_id] + base_ids + [tokenizer.eos_token_id]
        if len(base_ids) < max_length:
            base_ids = base_ids + [tokenizer.pad_token_id] * (max_length - len(base_ids))
        self.base_ids = base_ids

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        """Rotate the prompt slightly so successive steps see fresh input."""
        offset = index % self.max_length
        ids = list(self.base_ids)
        if offset and len(ids) > 4:
            ids = ids[offset:] + ids[:offset]
        labels = self.tokenizer.build_labels(ids)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_jsonl_dataset(
    path: str | Path,
    tokenizer,
    max_length: int = 1024,
    system: str | None = "You are Carina, a warm virtual companion.",
) -> CarinaSFTDataset:
    """Load a JSONL/Parquet of tokenised conversations into a :class:`CarinaSFTDataset`."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(str(p))
        records = table.to_pylist()
        # Tokenised rows carry ``input_ids`` / ``labels`` already.
        ds_records: list[dict] = []
        for r in records:
            ids = list(r["input_ids"])
            labels = list(r["labels"])
            if len(ids) > max_length:
                ids = ids[:max_length]
                labels = labels[:max_length]
            ds_records.append({"input_ids": ids, "labels": labels})
        return _dataset_from_tokenised(ds_records, tokenizer, max_length)
    records = []
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "input_ids" in rec and "labels" in rec:
                # Pre-tokenised record (e.g. emitted by build_dataset).
                ids = list(rec["input_ids"])
                labels = list(rec["labels"])
                if len(ids) > max_length:
                    ids = ids[len(ids) - max_length:]  # keep END (assistant reply)
                    labels = labels[len(labels) - max_length:]
                # Skip records with broken labels (truncation dropped all assistant tokens).
                if sum(1 for l in labels if l != -100) < 16:
                    continue
                records.append({"input_ids": ids, "labels": labels})
            elif "conversations" in rec:
                records.append(rec)
    if records and "input_ids" in records[0]:
        return _dataset_from_tokenised(records, tokenizer, max_length)
    return CarinaSFTDataset(records, tokenizer, max_length=max_length, system=system)


def _dataset_from_tokenised(
    records: list[dict],
    tokenizer,
    max_length: int,
) -> CarinaSFTDataset:
    """Wrap pre-tokenised ``{input_ids, labels}`` rows in a dataset."""
    ds = CarinaSFTDataset.__new__(CarinaSFTDataset)
    ds.records = records
    ds.tokenizer = tokenizer
    ds.max_length = max_length
    ds.system = None
    return ds


class PretrainDataset(Dataset):
    """Plain causal-LM dataset reading ``{"input_ids": [...]}`` JSONL records."""

    def __init__(self, path: Path, max_length: int) -> None:
        self.max_length = max_length
        self.records: list[list[int]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ids = obj.get("input_ids") or obj.get("ids") or []
                if isinstance(ids, list) and ids:
                    self.records.append([int(i) for i in ids])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        ids = self.records[index][: self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [0] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ids, dtype=torch.long)


class DummyPretrainDataset(Dataset):
    """Tiny in-memory pretrain dataset for smoke tests."""

    def __init__(self, tokenizer, num_samples: int = 8, max_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        base_text = "the quick brown fox jumps over the lazy dog " * 4
        base = tokenizer.encode(base_text)[: max_length]
        if len(base) < max_length:
            base = base + [tokenizer.pad_token_id] * (max_length - len(base))
        self.base = base

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        offset = index % self.max_length
        ids = list(self.base)
        if offset and len(ids) > 4:
            ids = ids[offset:] + ids[:offset]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ids, dtype=torch.long)


__all__ = ["CarinaSFTDataset", "DummySFTDataset", "load_jsonl_dataset",
           "PretrainDataset", "DummyPretrainDataset"]
