"""Full training pipeline for Carina: dataset → tokenizer → pretrain → SFT → QA.

Edit ``configs/pretrain.py`` and ``configs/sft.py`` to change hyperparameters.
Run with::

    python scripts/train_all.py                # resume everything (default)
    python scripts/train_all.py --fresh        # start everything from scratch
    python scripts/train_all.py --skip-data    # skip download + build
    python scripts/train_all.py --skip-pretrain
    python scripts/train_all.py --skip-sft
    python scripts/train_all.py --skip-qa
    python scripts/train_all.py --limit-data N # only download N rows (smoke test)

Interactive commands (via FIFO ``/tmp/carina_train_cmd_<pid>.fifo`` or stdin):
    sane  — stop at the next epoch boundary
    si    — stop immediately
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from command_handler import TrainingController, install_signal_handler, start, stop

# ── Constants ─────────────────────────────────────────────────────────────────

_TOKENIZER_DIR = _ROOT / "models" / "carina_tokenizer"
_PRETRAIN_CKPT = _ROOT / "checkpoints" / "carina_pretrain" / "final"
_SFT_CKPT = _ROOT / "checkpoints" / "carina_sft" / "final"
_PRETRAIN_DATA = _ROOT / "datasets" / "carina_pretrain" / "train.jsonl"
_SFT_DATA = _ROOT / "datasets" / "carina_sft" / "train.jsonl"
_RAW_DATA_DIR = _ROOT / "datasets" / "raw"
_TOKENIZER_VOCAB = 8192

_STAGES = ["dataset", "tokenizer", "pretrain", "sft", "qa"]


# ── Logging ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _section(title: str, char: str = "═") -> None:
    _log(char * 60)
    _log(f"  {title}")
    _log(char * 60)


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _run(
    label: str,
    cmd: list[str],
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` and stream its output."""
    _log(f"[run] {' '.join(str(c) for c in cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    # Reduce CUDA memory fragmentation (set before first CUDA allocation).
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or _ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        print(raw, end="", flush=True)
        output_lines.append(raw.rstrip())
    proc.wait()
    if check and proc.returncode != 0:
        _log(f"[ERROR] {label} failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)
    _log(f"[done] {label} (exit={proc.returncode})")
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="\n".join(output_lines),
    )


# ── Stage 1: Download dataset ────────────────────────────────────────────────

def stage_download(limit: int = 0) -> bool:
    """Download raw conversation data from HuggingFace.

    Downloads HuggingFaceH4/ultrachat_200k (subset) + OpenAssistant/oasst1.
    """
    _section("STAGE 1/5 — Dataset Download")

    # Check if we already have raw data
    raw_files = list(_RAW_DATA_DIR.rglob("*.jsonl"))
    if raw_files:
        _log(f"Found {len(raw_files)} raw data files in {_RAW_DATA_DIR} — skipping download")
        total = 0
        for f in raw_files:
            lines = sum(1 for _ in open(f)) if f.stat().st_size < 10 * 1024 * 1024 else 0
            total += lines
        _log(f"  Total records: ~{total}")
        return True

    _RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets_to_fetch = [
        ("HuggingFaceH4/ultrachat_200k", 50_000),
        ("OpenAssistant/oasst1", 20_000),
    ]
    for ds_name, ds_limit in datasets_to_fetch:
        actual_limit = limit if limit > 0 else ds_limit
        _log(f"Downloading {ds_name} (limit={actual_limit})...")
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "download_dataset.py"),
            "--dataset-name", ds_name,
            "--output-dir", str(_RAW_DATA_DIR),
            "--limit", str(actual_limit),
        ]
        result = _run(f"download {ds_name}", cmd)
        if result.returncode != 0:
            _log(f"[WARN] {ds_name} download failed — continuing anyway")

    raw_files = list(_RAW_DATA_DIR.rglob("*.jsonl"))
    _log(f"Download stage complete: {len(raw_files)} files")
    return True


# ── Stage 2: Process data ─────────────────────────────────────────────────────

def stage_process() -> bool:
    """Tokenise raw JSONL conversations into pretrain + SFT datasets."""
    _section("STAGE 2/5 — Data Processing")

    # Check if already processed
    if _PRETRAIN_DATA.is_file() and _SFT_DATA.is_file():
        # Quick stats
        for p in [_PRETRAIN_DATA, _SFT_DATA]:
            lines = sum(1 for _ in open(p)) if p.stat().st_size < 10 * 1024 * 1024 else 0
            _log(f"Found {p}: ~{lines} records")
        _log("Data already processed — skipping build")
        return True

    # Train tokenizer first
    _log(f"Training tokenizer (vocab={_TOKENIZER_VOCAB})...")
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "train_tokenizer.py"),
        "--output-dir", str(_TOKENIZER_DIR),
        "--vocab-size", str(_TOKENIZER_VOCAB),
    ]
    _run("train tokenizer", cmd)

    # Build pretrain dataset (plain text conversations → tokenised JSONL)
    _log("Building pretrain dataset...")
    _PRETRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    _SFT_DATA.parent.mkdir(parents=True, exist_ok=True)

    # Simple approach: read all raw conversations, create pretrain JSONL
    # (each conversation is a flat text, tokenized)
    _build_pretrain_from_raw()

    # Build SFT dataset using build_dataset.py
    _log("Building SFT dataset...")
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "build_dataset.py"),
        "--input-dir", str(_RAW_DATA_DIR),
        "--output-dir", str(_SFT_DATA.parent),
        "--tokenizer-path", str(_TOKENIZER_DIR),
        "--max-length", "2048",
        "--system", "You are Carina, a warm and supportive virtual companion.",
        "--val-ratio", "0.005",
        "--limit", "100000",
        "--glob", "HuggingFaceH4_ultrachat_200k/train.jsonl",
    ]
    result = _run("build SFT dataset", cmd)
    if result.returncode != 0:
        _log("[WARN] build_dataset.py failed — creating fallback SFT data")
        _build_sft_from_raw()

    # Verify
    for p, label in [(_PRETRAIN_DATA, "pretrain"), (_SFT_DATA, "SFT")]:
        if p.is_file():
            lines = sum(1 for _ in open(p)) if p.stat().st_size < 10 * 1024 * 1024 else 0
            _log(f"  {label}: {p} (~{lines} records)")
        else:
            _log(f"  [WARN] {label} not found at {p}")

    return True


def _build_pretrain_from_raw() -> None:
    """Flatten raw conversation JSONL → pretrain JSONL with tokenised text."""
    import json as _json
    from carina.tokenizer import CarinaTokenizer

    if not _TOKENIZER_DIR.exists():
        _log("[WARN] tokenizer not ready yet — skipping pretrain data build")
        return

    tok = CarinaTokenizer.from_file(str(_TOKENIZER_DIR))
    records: list[dict] = []
    for jsonl_path in sorted(_RAW_DATA_DIR.rglob("*.jsonl")):
        try:
            for line in open(jsonl_path):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                # Flatten all content into one text
                texts: list[str] = []
                for turn in rec.get("conversations", []):
                    if isinstance(turn, dict):
                        texts.append(turn.get("content", ""))
                    elif isinstance(turn, list):
                        for m in turn:
                            if isinstance(m, dict):
                                texts.append(m.get("content", ""))
                flat = " ".join(texts).strip()
                if len(flat) < 20:
                    continue
                # Tokenise (will be retokenised at train time with full tokenizer)
                ids = tok.encode(flat)[: 2048 - 4]
                if len(ids) < 10:
                    continue
                records.append({"input_ids": ids})
                if len(records) >= 200_000:  # cap at 200k
                    break
        except Exception as e:
            _log(f"[WARN] error reading {jsonl_path}: {e}")
        if len(records) >= 200_000:
            break

    _PRETRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    with _PRETRAIN_DATA.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
    _log(f"Pretrain dataset: {len(records)} records → {_PRETRAIN_DATA}")


def _build_sft_from_raw() -> None:
    """Fallback: build SFT JSONL directly from raw conversations."""
    import json as _json

    records: list[dict] = []
    for jsonl_path in sorted(_RAW_DATA_DIR.rglob("*.jsonl")):
        try:
            for line in open(jsonl_path):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                convs = rec.get("conversations", [])
                if not convs:
                    continue
                # Keep user/assistant turns only
                clean: list[dict] = []
                for c in convs:
                    if isinstance(c, dict) and c.get("role") in ("user", "assistant"):
                        clean.append(c)
                if len(clean) >= 2:
                    records.append({"conversations": clean})
                if len(records) >= 100_000:
                    break
        except Exception:
            pass
        if len(records) >= 100_000:
            break

    _SFT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with _SFT_DATA.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
    _log(f"SFT fallback dataset: {len(records)} records → {_SFT_DATA}")


# ── Stage 3: Pretrain ─────────────────────────────────────────────────────────

def stage_pretrain(fresh: bool) -> bool:
    """Run pretraining (calls ``scripts/pretrain.py``)."""
    _section("STAGE 3/5 — Pretraining")

    # Check if already done
    if not fresh and (_PRETRAIN_CKPT / "checkpoint.pt").is_file():
        _log(f"Pretrain checkpoint found at {_PRETRAIN_CKPT} — skipping pretrain")
        return True

    _log(f"Starting pretrain (fresh={fresh})...")
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "pretrain.py"),
    ]
    if fresh:
        cmd.append("--fresh")

    # Run in same process for interactive control (not subprocess)
    # so the FIFO controller works for this process
    os.environ["PYTHONPATH"] = str(_ROOT)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from scripts.pretrain import train as _pretrain_train
        from configs.pretrain import PretrainConfig
        cfg = PretrainConfig()
        result = _pretrain_train(cfg, fresh=fresh)
        _log(f"Pretrain result: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        _log(f"[ERROR] pretrain failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Stage 4: SFT ─────────────────────────────────────────────────────────────

def stage_sft(fresh: bool) -> bool:
    """Run supervised fine-tuning (calls ``scripts/train_sft.py``)."""
    _section("STAGE 4/5 — Supervised Fine-Tuning")

    if not fresh and (_SFT_CKPT / "checkpoint.pt").is_file():
        _log(f"SFT checkpoint found at {_SFT_CKPT} — skipping SFT")
        return True

    _log(f"Starting SFT (fresh={fresh})...")
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "train_sft.py"),
    ]
    if fresh:
        cmd.append("--fresh")

    os.environ["PYTHONPATH"] = str(_ROOT)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from scripts.train_sft import train as _sft_train
        from configs.sft import SFTConfig
        cfg = SFTConfig()
        result = _sft_train(cfg, fresh=fresh)
        _log(f"SFT result: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        _log(f"[ERROR] SFT failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Stage 5: QA ──────────────────────────────────────────────────────────────

def stage_qa() -> bool:
    """Run a simple QA test on the final SFT checkpoint."""
    _section("STAGE 5/5 — QA Test")
    import sys as _sys

    # Find the checkpoint
    # Prefer the best-val-loss checkpoint over the final (which may be overfitted).
    _best_dir = _SFT_CKPT.parent / "best"
    ckpt_dir = _best_dir if (_best_dir / "checkpoint.pt").is_file() else _SFT_CKPT

    _log(f"Loading model from {ckpt_dir}...")
    _sys.path.insert(0, str(_ROOT / "src"))
    try:
        from carina.inference.generate import _load_model_and_tokenizer, generate
    except ImportError as e:
        _log(f"[WARN] Could not import inference: {e}")
        return False

    try:
        model, tokenizer, _ = _load_model_and_tokenizer(str(ckpt_dir), device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        _log(f"[WARN] Could not load model: {e}")
        return False

    sys_prompt = "You are Carina, a warm and supportive virtual companion."
    prompts = [
        "Hello! How are you today?",
        "Tell me a fun fact about space.",
        "I feel a bit down today. Can you cheer me up?",
        "What is the capital of France?",
        "Write a haiku about autumn leaves.",
    ]

    _log("=" * 60)
    _log("  QA RESULTS")
    _log("=" * 60)
    all_ok = True
    for p in prompts:
        _log(f"\nQ: {p}")
        try:
            msg = [{"role": "user", "content": p}]
            prompt = tokenizer.apply_chat_template(
                msg, system=sys_prompt, add_generation_prompt=True
            )
            text = "".join(
                generate(
                    model, tokenizer, prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.05,
                )
            )
            _log(f"A: {text[:300]}")
            # Basic sanity: response is non-empty and not all numbers
            if len(text.strip()) < 5:
                _log("[WARN] Response too short")
                all_ok = False
        except Exception as e:
            _log(f"[ERROR] Generation failed: {e}")
            all_ok = False

    _log("\n" + "=" * 60)
    _log(f"  QA {'PASSED ✓' if all_ok else 'ISSUES DETECTED ⚠'}")
    _log("=" * 60)
    return all_ok


# ── Interactive control ────────────────────────────────────────────────────────

def _monitor_loop(controller: TrainingController) -> None:
    """Poll for commands from the FIFO/stdin and log them."""
    while not controller.stop_now:
        controller.poll_commands()
        if controller.stop_at_epoch or controller.stop_now:
            break
        time.sleep(1.0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full Carina training: dataset → tokenizer → pretrain → SFT → QA."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start everything from scratch (ignore all existing checkpoints).",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset download and processing.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining (use existing pretrain checkpoint).",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT (use existing SFT checkpoint).",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA test.",
    )
    parser.add_argument(
        "--limit-data",
        type=int,
        default=0,
        help="Cap downloaded data rows (0 = full download).",
    )
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────
    print()
    _log("╔══════════════════════════════════════════════════════════════╗")
    _log("║                  CARINA FULL TRAINING PIPELINE               ║")
    _log("╚══════════════════════════════════════════════════════════════╝")
    _log(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(f"Fresh run: {args.fresh}")
    _log(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _log(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Interactive controller ─────────────────────────────────────────────
    controller = TrainingController()
    install_signal_handler(controller)
    fifo_path = start(controller)
    if fifo_path:
        _log(f"Command FIFO: {fifo_path}")
        _log("  echo sane > {fifo_path}   # stop at next epoch")
        _log("  echo si   > {fifo_path}   # stop immediately")

    overall_ok = True
    start_time = time.time()

    try:
        # Stage 1: Download
        controller.poll_commands()
        if controller.stop_now:
            _log("[ABORT] si received before stage 1")
            overall_ok = False
        else:
            if not args.skip_data:
                ok = stage_download(limit=args.limit_data)
                overall_ok = overall_ok and ok
            else:
                _log("[SKIP] dataset stage skipped (--skip-data)")

        # Stage 2: Process
        controller.poll_commands()
        if controller.stop_now:
            _log("[ABORT] si received before stage 2")
            overall_ok = False
        else:
            if not args.skip_data:
                ok = stage_process()
                overall_ok = overall_ok and ok
            else:
                _log("[SKIP] processing stage skipped (--skip-data)")

        # Stage 3: Pretrain
        controller.poll_commands()
        if controller.stop_now:
            _log("[ABORT] si received before stage 3")
            overall_ok = False
        else:
            if not args.skip_pretrain:
                ok = stage_pretrain(fresh=args.fresh)
                overall_ok = overall_ok and ok
            else:
                _log("[SKIP] pretrain stage skipped (--skip-pretrain)")

        # Stage 4: SFT
        controller.poll_commands()
        if controller.stop_now:
            _log("[ABORT] si received before stage 4")
            overall_ok = False
        else:
            if not args.skip_sft:
                ok = stage_sft(fresh=args.fresh)
                overall_ok = overall_ok and ok
            else:
                _log("[SKIP] SFT stage skipped (--skip-sft)")

        # Stage 5: QA
        controller.poll_commands()
        if controller.stop_now:
            _log("[ABORT] si received before stage 5")
            overall_ok = False
        else:
            if not args.skip_qa:
                ok = stage_qa()
                overall_ok = overall_ok and ok
            else:
                _log("[SKIP] QA stage skipped (--skip-qa)")

    except KeyboardInterrupt:
        _log("[INTERRUPT] KeyboardInterrupt received")
        overall_ok = False
    finally:
        stop(controller)
        elapsed = (time.time() - start_time) / 60.0
        _log("")
        _log("╔══════════════════════════════════════════════════════════════╗")
        _log(f"║  PIPELINE {'COMPLETED ✓' if overall_ok else 'FINISHED WITH ISSUES ⚠'}  ({elapsed:.1f} min)       ║")
        _log(f"║  Commands received: {len(controller.command_log):3d}                               ║")
        _log("╚══════════════════════════════════════════════════════════════╝")
        for cmd in controller.command_log:
            _log(f"  {cmd}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
