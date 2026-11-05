"""Memory-lean pretrain corpus: tokenize once into a flat int32 numpy array.

Avoids PretrainDataset's double-copy of 22M python ints (~2GB+ RAM).
Layout: one flat token stream (docs separated by eos), sliced into
non-overlapping windows of seq_len+1 at train time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carina.config import DataConfig, TokenizerConfig
from carina.tokenizer import ensure_tokenizer


def build_cache(src: Path, dst: Path, tokenizer_dir: Path, hf_id: str) -> Path:
    if dst.exists():
        print(f"cache exists: {dst}")
        return dst
    tok = ensure_tokenizer(hf_id, tokenizer_dir, False)
    eos = int(tok.eos_token_id)
    chunks: list[np.ndarray] = []
    total = 0
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or obj.get("story") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tok.encode(text.strip())[: int(DataConfig.lccc_max_chars) // 2]
            chunks.append(np.asarray(ids + [eos], dtype=np.int32))
            total += len(ids) + 1
            if len(chunks) % 100000 == 0:
                print(f"  {total:,} tokens", flush=True)
    flat = np.concatenate(chunks)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, flat)
    print(f"wrote {dst} tokens={len(flat):,} docs={len(chunks):,}")
    return dst


if __name__ == "__main__":
    out = Path("/tmp/carina_cache/tokens.npy")
    build_cache(
        Path(DataConfig.pretrain_processed_saved_path),
        out,
        Path(TokenizerConfig.tokenizer_dir),
        TokenizerConfig.hf_id,
    )
