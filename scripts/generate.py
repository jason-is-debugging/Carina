"""Generation: greedy decode with KV-cache switch, config-driven."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from carina.config import MiscConfig, ModelConfig, TokenizerConfig
from carina.config import TrainingConfig
from carina.model.carina_casual_model import CarinaCausalModel
from carina.tokenizer import ensure_tokenizer
from carina.utils import get_device, set_seed
from carina.utils.logger import make_logger


def parse_args() -> argparse.Namespace:
    """Parse prompt + generation options."""
    parser = argparse.ArgumentParser(description="Generate with Carina")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true",
                        help="disable KV-cache (full forward each step)")
    return parser.parse_args()


def build_model(vocab_size: int, ign: int) -> CarinaCausalModel:
    """Build CarinaCausalModel from ModelConfig (same as train)."""
    return CarinaCausalModel(
        vocab_size=vocab_size,
        d_model=ModelConfig.d_model,
        d_hidden=ModelConfig.d_ff,
        n_heads=ModelConfig.n_heads,
        max_loop_times=ModelConfig.max_loop_times,
        lora_reusage=ModelConfig.lora_reusage,
        ranks=tuple(ModelConfig.ranks),
        dropout=ModelConfig.dropout,
        tie=ModelConfig.tie_embeddings,
        ignore_index=ign,
    )


def main() -> None:
    """Load best checkpoint and greedy-generate from prompt."""
    args = parse_args()
    set_seed(MiscConfig.seed)
    device = get_device()
    save_dir = Path(TrainingConfig.model_saved_dir)
    logger = make_logger("generate")
    tok = ensure_tokenizer(TokenizerConfig.hf_id,
                           TokenizerConfig.tokenizer_dir,
                           TokenizerConfig.trust_remote_code)
    ign = int(TokenizerConfig.label_ignore_token)
    model = build_model(len(tok), ign).to(device).to(torch.bfloat16)
    try:
        state = torch.load(save_dir / "best.pt", map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        logger.info("loaded best checkpoint")
    except FileNotFoundError:
        logger.warning("no checkpoint, using random weights")
    prompt_ids = tok.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tok.eos_token_id
    eos_id = int(eos_id) if eos_id is not None else None
    start = time.time()
    out = model.generate(input_ids, args.max_new_tokens,
                         eos_id=eos_id, use_cache=not args.no_cache)
    elapsed = max(time.time() - start, 1e-6)
    new_ids = out[0].tolist()[len(prompt_ids):]
    text = tok.decode(out[0].tolist())
    logger.info(f"text={text[:500]}")
    logger.info(f"new_tokens={len(new_ids)} "
                f"tok_s={len(new_ids) / elapsed:.1f} "
                f"use_cache={not args.no_cache}")
    print(text)


if __name__ == "__main__":
    main()
