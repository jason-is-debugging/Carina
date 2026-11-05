"""Prepare corpora: download raw -> process -> save, driven by DataConfig."""
from __future__ import annotations

import argparse

from carina.config import DataConfig, MiscConfig, PersonaConfig
from carina.datasets import build_all


def main() -> None:
    """Run Muice + LCCC corpus build per DataConfig."""
    parser = argparse.ArgumentParser(description="Build Carina corpora")
    parser.add_argument(
        "--only",
        nargs="+",
        default=["muice", "lccc"],
        help="subset of sources to build",
    )
    args = parser.parse_args()

    outs = build_all(
        corpora_dir=DataConfig.corpora_dir,
        lccc_hf_id=DataConfig.lccc_hf_id,
        muice_hf_id=DataConfig.muice_hf_id,
        muice_orig_names=PersonaConfig.muice_orig_names,
        persona_zh=PersonaConfig.name_zh,
        persona_en=PersonaConfig.name_en,
        lccc_pretrain_samples=DataConfig.lccc_pretrain_samples,
        lccc_min_chars=DataConfig.lccc_min_chars,
        lccc_max_chars=DataConfig.lccc_max_chars,
        lccc_sft_samples=DataConfig.lccc_sft_samples,
        muice_upsample=DataConfig.muice_upsample,
        seed=MiscConfig.seed,
        only=tuple(args.only),
    )
    for name, path in outs.items():
        print(f"{name} -> {path}")


if __name__ == "__main__":
    main()
