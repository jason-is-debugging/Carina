"""Gradio web demo for Carina chat.

Launches a local chat UI. ``gradio`` is the only optional dependency;
importing this module will fail loudly if it is not installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import CarinaConfig
from ..model import CarinaForCausalLM
from ..tokenizer import CarinaTokenizer
from .generate import generate

DEFAULT_SYSTEM = "You are Carina, a warm, supportive virtual companion."


def _load_for_demo(checkpoint):
    checkpoint = Path(checkpoint)
    tokenizer_path = checkpoint / "tokenizer"
    if not tokenizer_path.is_dir():
        tokenizer_path = checkpoint
    tokenizer = CarinaTokenizer.from_file(tokenizer_path)
    config_path = checkpoint / "carina_config.json"
    if config_path.is_file():
        config = CarinaConfig.load(config_path)
    else:
        config = CarinaConfig(vocab_size=tokenizer.vocab_size)
    model = CarinaForCausalLM(config)
    state = checkpoint / "model.pt"
    if not state.is_file():
        state = checkpoint / "checkpoint" / "checkpoint.pt"
    if state.is_file():
        raw = torch.load(str(state), map_location="cpu", weights_only=False)
        sd = raw.get("model", raw) if isinstance(raw, dict) else raw
        model.load_state_dict(sd, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer, config, device


def build_demo(checkpoint, share: bool = False, server_port: int | None = None):
    """Build and return a :class:`gradio.Blocks` demo object.

    Args:
        checkpoint: Path to a saved Carina checkpoint.
        share: Whether to expose a public ``gradio.live`` link.
        server_port: Optional port to bind.
    """
    import gradio as gr

    model, tokenizer, _config, device = _load_for_demo(checkpoint)

    def respond(message, history):
        messages = list(history or []) + [{"role": "user", "content": message}]
        prompt = tokenizer.apply_chat_template(messages, system=DEFAULT_SYSTEM)
        return "".join(
            generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                stream=True,
                device=device,
            )
        )

    demo = gr.ChatInterface(
        fn=respond,
        title="Carina — virtual companion",
        description=(
            "A small, friendly language model. Responses are generated "
            "locally from a Carina checkpoint."
        ),
    )
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the Carina Gradio demo.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/carina_sft/final")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    if not Path(args.checkpoint).is_dir():
        print(f"Checkpoint directory {args.checkpoint} not found.")
        return 1
    demo = build_demo(args.checkpoint, share=args.share, server_port=args.port)
    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_demo", "main"]
