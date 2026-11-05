"""Corpus pipeline: download raw -> process -> saved processed JSONL.

No config import here; all values are passed in as arguments.
The script entry (scripts/prepare_data.py) reads DataConfig and calls in.
"""
from __future__ import annotations

from pathlib import Path

from .lccc import (
    download_lccc_dataset,
    process_lccc_pretrain,
    process_lccc_sft,
)
from .merge import merge_sft
from .muice import (
    download_muice_dataset,
    process_muice_dataset,
)
from .pretrain import download_pretrain_dataset, process_pretrain_dataset
from .sft import download_finetune_dataset, process_finetune_dataset


def build_pretrain_corpus(
    source: str,
    raw_path: str | Path,
    processed_path: str | Path,
    max_samples: int = 2000,
    min_chars: int = 16,
) -> Path:
    """Download + clean pretrain corpus, return processed path."""
    raw = Path(raw_path)
    processed = Path(processed_path)
    download_pretrain_dataset(source, raw, max_samples=max_samples)
    process_pretrain_dataset(raw, processed, min_chars=min_chars)
    return processed


def build_finetune_corpus(
    source: str,
    raw_path: str | Path,
    processed_path: str | Path,
    max_samples: int = 2000,
    min_chars: int = 8,
) -> Path:
    """Download + clean SFT corpus, return processed path."""
    raw = Path(raw_path)
    processed = Path(processed_path)
    download_finetune_dataset(source, raw, max_samples=max_samples)
    process_finetune_dataset(raw, processed, min_chars=min_chars)
    return processed


def build_all(
    corpora_dir: str | Path,
    lccc_hf_id: str,
    muice_hf_id: str,
    muice_orig_names: tuple[str, ...],
    persona_zh: str,
    persona_en: str,
    lccc_pretrain_samples: int = 70000,
    lccc_min_chars: int = 8,
    lccc_max_chars: int = 512,
    lccc_sft_samples: int = 5000,
    muice_upsample: float = 3.0,
    seed: int = 42,
    only: tuple[str, ...] = ("muice", "lccc"),
) -> dict[str, Path]:
    """Full two-source pipeline: download+process each, then merge the SFT.

    Returns:
        dict of {name: processed path} incl. "pretrain_merged",
        "finetune_merged". A missing source is warned about and skipped
        rather than aborting the run.
    """
    corpora = Path(corpora_dir)
    corpora.mkdir(parents=True, exist_ok=True)
    outs: dict[str, Path] = {}
    sft_parts: list[Path] = []
    sft_weights: list[float] = []

    if "muice" in only:
        try:
            raw = corpora / "muice_raw.jsonl"
            proc = corpora / "muice_processed.jsonl"
            download_muice_dataset(muice_hf_id, raw)
            process_muice_dataset(raw, proc, muice_orig_names,
                                  persona_zh, persona_en)
            outs["muice"] = proc
            sft_parts.append(proc)
            sft_weights.append(muice_upsample)
            print(f"[pipeline] muice ok -> {proc}")
        except Exception as e:
            print(f"[pipeline] muice SKIP: {type(e).__name__}: {e!s:.300s}")

    if "lccc" in only:
        try:
            raw = corpora / "lccc_raw.jsonl"
            download_lccc_dataset(lccc_hf_id, raw)
            pretrain = corpora / "lccc_pretrain.jsonl"
            process_lccc_pretrain(raw, pretrain, lccc_min_chars,
                                  lccc_max_chars, lccc_pretrain_samples)
            outs["pretrain_merged"] = pretrain
            sft = corpora / "lccc_sft.jsonl"
            process_lccc_sft(raw, sft, lccc_min_chars,
                             lccc_sft_samples, skip=lccc_pretrain_samples)
            outs["lccc_sft"] = sft
            sft_parts.append(sft)
            sft_weights.append(1.0)
            print(f"[pipeline] lccc ok -> {pretrain} + {sft}")
        except Exception as e:
            print(f"[pipeline] lccc SKIP: {type(e).__name__}: {e!s:.300s}")

    if sft_parts:
        merged = corpora / "finetune_merged.jsonl"
        merge_sft(sft_parts, sft_weights, merged, seed=seed)
        outs["finetune_merged"] = merged
        print(f"[pipeline] merged SFT -> {merged} "
              f"({len(sft_parts)} sources)")
    return outs
