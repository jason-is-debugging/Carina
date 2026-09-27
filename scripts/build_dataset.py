"""Build a Carina SFT dataset from raw JSONL conversations.

Reads each ``.jsonl`` line shaped like
``{"conversations": [{"role": ..., "content": ...}, ...]}``, applies the
Carina chat template, tokenises with the chosen tokenizer, and writes
``train.jsonl`` plus ``val.jsonl`` (when ``--val-ratio > 0``) under
``<output-dir>``.

Memory-efficient: records are tokenised and written one-by-one so
memory usage stays constant regardless of dataset size.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from carina.tokenizer import CarinaTokenizer


def _load_records(input_dir: Path, glob_pattern: str = "**/*.jsonl") -> list[dict]:
    records: list[dict] = []
    for path in sorted(input_dir.glob(glob_pattern)):
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("conversations"):
                    records.append(rec)
    return records


def _tokenize_and_write(
    records: list[dict],
    tokenizer: CarinaTokenizer,
    max_length: int,
    system: str,
    out_path: Path,
) -> dict:
    """Tokenise ``records`` one-by-one and stream-write to ``out_path``."""
    n = 0
    total_len = 0
    max_len = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            convo = rec.get("conversations", [])
            messages: list[dict] = []
            if system and not any(m.get("role") == "system" for m in convo):
                messages.append({"role": "system", "content": system})
            for turn in convo:
                role = turn.get("role")
                if role not in ("system", "user", "assistant"):
                    continue
                messages.append({"role": role, "content": turn.get("content", "")})
            if not messages:
                continue
            # If the entire conversation exceeds max_length, truncate the
            # OLDEST turns from the START so the most recent (assistant)
            # reply is always kept.  We do this on text before encoding
            # so the role tags / EOS tokens are never broken.
            rendered = tokenizer.apply_chat_template(messages, system=system, add_generation_prompt=False)
            ids = tokenizer.encode(rendered)
            if len(ids) > max_length:
                # Iteratively drop the first (oldest) user+assistant pair
                # until the result fits.  Each pair has: USER-tag + content + EOS + ASSISTANT-tag + content + EOS.
                # We remove text from the oldest USER content onwards.
                for _ in range(50):  # safety cap
                    # Find the second USER tag (skip BOS + any initial USER).
                    second_user_pos = rendered.find("<|user|>", rendered.find("<|user|>") + 1)
                    if second_user_pos < 0:
                        break
                    # Re-render messages without the prefix up to that USER tag.
                    prefix_trimmed = rendered[second_user_pos:]
                    ids = tokenizer.encode(prefix_trimmed)
                    if len(ids) <= max_length:
                        rendered = prefix_trimmed
                        break
            # Final safety: if still over limit, truncate from the START
            # (keep the last assistant reply intact).
            if len(ids) > max_length:
                ids = ids[len(ids) - max_length:]
            if len(ids) < max_length:
                ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
            labels = tokenizer.build_labels(ids)
            fh.write(json.dumps({"input_ids": ids, "labels": labels}, ensure_ascii=False) + "\n")
            total_len += len(ids)
            max_len = max(max_len, len(ids))
            n += 1
    return {
        "num_samples": n,
        "avg_seq_len": round(total_len / max(1, n), 1),
        "max_seq_len": max_len,
    }


def build(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
    val_ratio: float = 0.02,
    max_length: int = 1024,
    system: str = "You are Carina, a warm and supportive virtual companion.",
    seed: int = 0,
    limit: int = 0,
    glob_pattern: str = "**/*.jsonl",
) -> dict:
    """Build the train/val split and write it under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CarinaTokenizer.from_file(tokenizer_path)

    # Load records matching glob, shuffle, limit, split.
    all_records = _load_records(input_dir, glob_pattern=glob_pattern)
    rng = random.Random(seed)
    rng.shuffle(all_records)
    if limit > 0:
        all_records = all_records[:limit]
    n_val = int(len(all_records) * val_ratio)
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    # Stream tokenise and write (constant memory).
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    train_stats = _tokenize_and_write(train_records, tokenizer, max_length, system, train_path)
    val_stats = _tokenize_and_write(val_records, tokenizer, max_length, system, val_path)

    return {
        "train": train_stats,
        "val": val_stats,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Carina SFT dataset.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/carina_sft"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("models/carina_tokenizer"))
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--system",
        type=str,
        default="You are Carina, a warm and supportive virtual companion.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of records to process (0 = unlimited).",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="**/*.jsonl",
        help="Glob pattern relative to --input-dir to select source files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = build(
        args.input_dir,
        args.output_dir,
        args.tokenizer_path,
        val_ratio=args.val_ratio,
        max_length=args.max_length,
        system=args.system,
        seed=args.seed,
        limit=args.limit,
        glob_pattern=args.glob_pattern,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
