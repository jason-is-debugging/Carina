"""Download an external companion/assistant dataset for Carina.

Supported datasets (any of the HuggingFace IDs below)::

    open-orca/SlimOrca          — small, distilled from OpenOrca
    HuggingFaceH4/ultrachat_200k — multi-turn English dialogue
    OpenAssistant/oasst1        — OASST1 conversation tree

The script writes the downloaded splits as ``.jsonl`` files under
``<output-dir>/<dataset-name>/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORTED_DATASETS = {
    "open-orca/SlimOrca": {"split": "train"},
    "HuggingFaceH4/ultrachat_200k": {"split": "train_sft"},
    "OpenAssistant/oasst1": {"split": "train"},
}


def _coerce(record: dict) -> dict | None:
    """Normalise a raw record into ``{"conversations": [...]}`` shape."""
    if "conversations" in record and isinstance(record["conversations"], list):
        return record
    if "messages" in record and isinstance(record["messages"], list):
        return {"conversations": record["messages"]}
    if "prompt" in record and "response" in record:
        return {
            "conversations": [
                {"role": "user", "content": str(record["prompt"])},
                {"role": "assistant", "content": str(record["response"])},
            ]
        }
    return None


def download_dataset(dataset_name: str, output_dir: Path, limit: int = 0) -> Path:
    """Download ``dataset_name`` into ``output_dir``.

    Args:
        dataset_name: A key from :data:`SUPPORTED_DATASETS`.
        output_dir: Root directory; ``<output_dir>/<dataset_name>/`` is
            created and ``train.jsonl`` is written inside it.
        limit: Optional cap on the number of records.

    Returns:
        The path to the resulting ``train.jsonl``.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; supported: {list(SUPPORTED_DATASETS)}"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The `datasets` package is required: pip install datasets") from exc
    cfg = SUPPORTED_DATASETS[dataset_name]
    target_dir = output_dir / dataset_name.replace("/", "_")
    target_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = target_dir / "train.jsonl"
    ds = load_dataset(dataset_name, split=cfg["split"], streaming=False)
    n = 0
    with jsonl_path.open("w", encoding="utf-8") as fh:
        from tqdm import tqdm

        for record in tqdm(ds, desc=f"download {dataset_name}"):
            coerced = _coerce(dict(record))
            if coerced is None:
                continue
            fh.write(json.dumps(coerced, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    print(f"Saved {n} records → {jsonl_path}")
    return jsonl_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a companion dataset.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    download_dataset(args.dataset_name, args.output_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
