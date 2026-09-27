"""Carina text generation with KV-cache support."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import CarinaConfig
from ..model import CarinaForCausalLM
from ..tokenizer import CarinaTokenizer
from .sampling import apply_repetition_penalty, sample_top_k_top_p
from .streaming import TextStreamer


def _load_model_and_tokenizer(
    checkpoint_dir: str | Path,
    device: torch.device | str | None = None,
):
    """Load a model + tokenizer pair from a saved checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer_path = checkpoint_dir / "tokenizer"
    if not tokenizer_path.is_dir():
        tokenizer_path = checkpoint_dir
    tokenizer = CarinaTokenizer.from_file(tokenizer_path)
    config_path = checkpoint_dir / "carina_config.json"
    if config_path.is_file():
        config = CarinaConfig.load(config_path)
    else:
        config = CarinaConfig(vocab_size=tokenizer.vocab_size)
    model = CarinaForCausalLM(config)
    state_path = checkpoint_dir / "model.pt"
    if not state_path.is_file():
        state_path = checkpoint_dir / "checkpoint" / "checkpoint.pt"
    if state_path.is_file():
        raw = torch.load(str(state_path), map_location="cpu", weights_only=False)
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        model.load_state_dict(state, strict=False)
    model.eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    return model, tokenizer, config


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    stream: bool = False,
    eos_token_id: int | None = None,
    device=None,
):
    """Generate ``max_new_tokens`` of text from ``prompt``.

    Args:
        model: A :class:`CarinaForCausalLM`.
        tokenizer: The matching :class:`CarinaTokenizer`.
        prompt: The prompt text. The role-tagged chat template is
            applied by callers (this function operates on raw ids).
        max_new_tokens: Maximum tokens to generate.
        temperature: Softmax temperature; ``0`` → greedy decoding.
        top_p: Nucleus sampling probability mass.
        top_k: Top-k cutoff (``0`` disables).
        repetition_penalty: Multiplicative penalty for repeated tokens.
        stream: If ``True`` the function yields one ``str`` per token
            batch; otherwise it yields the full decoded text once.
        eos_token_id: Override EOS id; defaults to ``tokenizer.eos``.
        device: Move inputs to this device before forwarding.

    Returns:
        Either the generated text (when ``stream=False``) or a ``str``
        per emitted chunk via ``yield``.
    """
    target_device = device or next(model.parameters()).device
    input_ids_list = tokenizer.encode(prompt)
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    if not input_ids_list:
        input_ids_list = [tokenizer.bos_token_id]
    generated: list[int] = []
    past = None
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None
    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                cur = torch.tensor([input_ids_list], dtype=torch.long, device=target_device)
            else:
                last = generated[-1]
                cur = torch.tensor([[last]], dtype=torch.long, device=target_device)
            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
            )
            logits = out["logits"][:, -1, :]
            past = out["past_key_values"]
            if repetition_penalty != 1.0:
                history = input_ids_list + generated
                history_t = torch.tensor(
                    [history], dtype=torch.long, device=logits.device
                )
                logits = apply_repetition_penalty(logits, history_t, repetition_penalty)
            next_token = sample_top_k_top_p(
                logits, top_k=top_k, top_p=top_p, temperature=temperature
            )
            token_id = int(next_token.item())
            generated.append(token_id)
            if streamer is not None:
                text = streamer.put([token_id])
                if text:
                    yield text
            if eos is not None and token_id == eos:
                break
    if streamer is not None:
        tail = streamer.end()
        if tail:
            yield tail
    if not stream:
        yield tokenizer.decode(generated)


def chat(model, tokenizer, messages, system=None, **kwargs) -> str:
    """One-shot chat helper: render ``messages``, generate, return text."""
    prompt = tokenizer.apply_chat_template(messages, system=system)
    return "".join(generate(model, tokenizer, prompt, **kwargs))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with Carina.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/carina_sft/final")
    parser.add_argument("--prompt", type=str, default="Hello!")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model, tokenizer, _ = _load_model_and_tokenizer(args.checkpoint)
    messages = [{"role": "user", "content": args.prompt}]
    pieces: list[str] = []
    for chunk in generate(
        model,
        tokenizer,
        tokenizer.apply_chat_template(messages, system=args.system, add_generation_prompt=True),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stream=args.stream,
    ):
        if args.stream:
            print(chunk, end="", flush=True)
        pieces.append(chunk)
    if not args.stream:
        print("".join(pieces))
    print()
    return 0


__all__ = ["_load_model_and_tokenizer", "chat", "generate", "main"]
