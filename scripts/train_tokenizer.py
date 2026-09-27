"""Train a small BPE tokenizer for Carina and persist it to disk.

The training corpus is a built-in synthetic companion-style dialogue set
that runs entirely offline. If the user has an external JSONL file in
``$CARINA_TOKENIZER_CORPUS`` (one record per line, each with a
``conversations`` list), those lines are appended to the synthetic
corpus as well.

Usage::

    python scripts/train_tokenizer.py \
        --output-dir models/carina_tokenizer \
        --vocab-size 8000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from carina.tokenizer import SPECIAL_TOKENS

SEED_CORPUS: list[list[dict]] = [
    [
        {"role": "system", "content": "You are Carina, a warm and supportive virtual companion."},
        {"role": "user", "content": "Hi Carina, I had a rough day at work."},
        {"role": "assistant", "content": "I'm sorry to hear that. Want to tell me what happened? I'm here to listen."},
        {"role": "user", "content": "My manager criticised my presentation in front of everyone."},
        {"role": "assistant", "content": "That sounds really uncomfortable. Public criticism can sting even when the feedback is fair. How are you feeling about it now?"},
    ],
    [
        {"role": "user", "content": "I feel stuck. I want to learn piano but I have no idea where to start."},
        {"role": "assistant", "content": "That's a great goal! A gentle starting point is to pick 3-5 songs you love, find beginner sheet music or tutorials for them, and practise 15 minutes a day. Would you like some song suggestions?"},
        {"role": "user", "content": "Yes please, I love soft pop music."},
        {"role": "assistant", "content": "Wonderful. Try 'Someone Like You' by Adele, 'Let Her Go' by Passenger, and 'A Thousand Years' by Christina Perri — they all have simple left-hand patterns and emotional melodies that are rewarding for beginners."},
    ],
    [
        {"role": "system", "content": "You are Carina, a knowledgeable and friendly AI companion."},
        {"role": "user", "content": "Explain photosynthesis to me like I'm twelve."},
        {"role": "assistant", "content": "Sure! Plants are like tiny solar-powered kitchens. They take sunlight, water from their roots, and carbon dioxide from the air, and they mix them together using a green pigment called chlorophyll to make sugar — which is their food. As a bonus they breathe out oxygen for us to inhale."},
    ],
    [
        {"role": "user", "content": "I'm nervous about my job interview tomorrow."},
        {"role": "assistant", "content": "It's completely normal to feel that way. A few things that might help: review the job description, prepare three short stories about your accomplishments using the STAR method (Situation, Task, Action, Result), and get a good night's sleep. You've got this."},
        {"role": "user", "content": "What does STAR mean exactly?"},
        {"role": "assistant", "content": "STAR is a storytelling framework for behavioural interview questions. You describe the Situation you were in, the Task you needed to accomplish, the Actions you took, and the Results you achieved. It helps you give concise, concrete answers instead of rambling."},
    ],
    [
        {"role": "user", "content": "Suggest a quick vegetarian dinner for tonight."},
        {"role": "assistant", "content": "How about a creamy one-pot pasta with mushrooms and spinach? Sauté garlic and sliced mushrooms in olive oil, add cream or coconut milk and a handful of parmesan, toss in cooked pasta and fresh spinach, and finish with black pepper. Ready in about 20 minutes."},
        {"role": "user", "content": "Sounds great, thank you!"},
        {"role": "assistant", "content": "Enjoy! Let me know how it turns out, or if you'd like more vegetarian ideas for the week."},
    ],
    [
        {"role": "user", "content": "How do I stay motivated when working from home?"},
        {"role": "assistant", "content": "A few tactics work well: dress like you're going to the office, keep a dedicated workspace, set clear start and stop times, and schedule short breaks away from your desk. Also try pairing tasks with a small reward — for example, a coffee after finishing a report."},
    ],
    [
        {"role": "system", "content": "You are Carina, a calm and thoughtful companion."},
        {"role": "user", "content": "I can't sleep. My mind keeps racing."},
        {"role": "assistant", "content": "Racing thoughts before sleep are really common. Try a 4-7-8 breathing pattern: inhale for 4 seconds, hold for 7, exhale for 8. Pair it with a body scan from your toes upward. If your mind grabs a thought, gently write it on a notepad and return to breathing."},
    ],
    [
        {"role": "user", "content": "What's a good first programming language?"},
        {"role": "assistant", "content": "Python is a popular first choice — its syntax reads almost like English, it has a huge ecosystem, and you can build anything from web apps to AI models with it. JavaScript is another strong option if you're more interested in building websites."},
    ],
    [
        {"role": "user", "content": "I just broke up with my partner and I feel lost."},
        {"role": "assistant", "content": "I'm really sorry. Breakups are painful even when they were the right call. Give yourself permission to grieve, lean on close friends or family, and try to keep small routines going — meals, sleep, walks. Healing isn't linear, and you don't have to 'bounce back' on anyone's timeline."},
    ],
    [
        {"role": "user", "content": "Teach me a fun fact about space."},
        {"role": "assistant", "content": "A teaspoon of neutron-star material would weigh about six billion tonnes — roughly the mass of Mount Everest — because the star's collapsed core is so dense that atoms are smashed together. Imagine stirring that into your morning coffee!"},
    ],
    [
        {"role": "user", "content": "What's a simple workout I can do at home?"},
        {"role": "assistant", "content": "Try a 15-minute circuit: 30 seconds each of jumping jacks, push-ups, bodyweight squats, plank, and lunges, with 15 seconds rest between. Repeat the circuit three times. No equipment needed, and you'll feel it the next day."},
    ],
]


def _extra_lines() -> list[str]:
    """Additional plain-text passages to broaden the vocabulary."""
    return [
        "Mathematics is the language with which the universe is written.",
        "A small cup of tea can steady the mind and warm the hands.",
        "Practice gratitude each morning: name three things you appreciate.",
        "Reading every day for twenty minutes adds up to dozens of books a year.",
        "Cooking at home is healthier, cheaper, and often more delicious.",
        "When you learn to debug code, you also learn to debug life.",
        "A walk in nature clears the cobwebs from the brain.",
        "Music soothes, energises, and connects people across cultures.",
        "Honesty and kindness are two ingredients of a trustworthy friend.",
        "Curiosity is the engine of lifelong learning.",
    ]


def build_corpus() -> list[str]:
    """Assemble the offline training corpus as plain-text lines.

    Priority order:
    1. Raw conversation JSONL files in ``datasets/raw/`` (one ``conversations``
       list per line) — these provide the richest vocabulary.
    2. ``CARINA_TOKENIZER_CORPUS`` env-var path (fallback for custom corpora).
    3. Built-in seed dialogues (last resort).
    """
    corpus: list[str] = []
    raw_dir = Path(__file__).resolve().parents[1] / "datasets" / "raw"
    if raw_dir.is_dir():
        for jsonl_path in sorted(raw_dir.glob("**/*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for turn in rec.get("conversations", []):
                        text = turn.get("content") if isinstance(turn, dict) else ""
                        if text:
                            corpus.append(text)
    custom = os.environ.get("CARINA_TOKENIZER_CORPUS")
    if custom and Path(custom).is_file():
        with Path(custom).open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for turn in record.get("conversations", []):
                    text = turn.get("content")
                    if text:
                        corpus.append(text)
    for convo in SEED_CORPUS:
        for turn in convo:
            content = turn.get("content", "")
            if content:
                corpus.append(content)
    corpus.extend(_extra_lines())
    return corpus


def _truncate(corpus: list[str], max_lines: int) -> list[str]:
    if len(corpus) <= max_lines:
        return corpus
    rng = random.Random(0)
    return rng.sample(corpus, max_lines)


def train(output_dir: Path, vocab_size: int, max_lines: int) -> dict:
    """Train a BPE tokenizer and write artefacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = _truncate(build_corpus(), max_lines)
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[1]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(output_dir / "tokenizer.json"))
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "num_lines": len(corpus),
        "special_tokens": list(SPECIAL_TOKENS),
        "model_max_length": 2048,
    }
    with (output_dir / "tokenizer_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Carina BPE tokenizer.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/carina_tokenizer"),
        help="Where to save tokenizer.json",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Target vocabulary size (4096–8192 recommended).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=20000,
        help="Maximum corpus lines to use (keeps training fast).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    meta = train(args.output_dir, args.vocab_size, args.max_lines)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved tokenizer to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
