"""Central experiment configuration.

All hyperparameters live here as plain class attributes.
Submodules must NOT import this file; values are injected via
constructor / function arguments (see README design decisions).
"""
from pathlib import Path


class ProjectConfig:
    """Global project config: shared across all experiments."""

    # --- project ---
    project_name = "carina"
    # Path(__file__) -> src/carina/config.py
    project_abs_path: Path = Path(__file__).parent.parent.parent.resolve()
    project_saved_path: Path = project_abs_path / "saved"
    datasets_saved_path: Path = project_abs_path / "datasets"


class MiscConfig:
    """Single-route config: shared decoder + per-round LoRA + PonderLM."""

    # --- reproducibility ---
    seed = 42

class PersonaConfig:
    """Assistant identity: replaces source-dataset role names."""

    name_en = "Carina"
    name_zh = "柳如烟"
    # Original Muice persona names: Chinese ones are replaced with name_zh,
    # ASCII ones with name_en.
    muice_orig_names = ("沐雪", "沐沐", "沐妮卡", "慕雪", "Muice", "muxue", "MUXUE")

class DataConfig:

    # --- loader ---
    # NOTE: the data is short (SFT p99=160, pretrain max=318), so the window
    # is tightened to 256 and pack=True concatenates several docs into one
    # sequence (eos separated, almost no padding); the freed memory goes to
    # widening the model.
    # Measured with real data: B3/d1024/ff4096/L16 peaks at 4.70GB plus
    # ~0.6GB resident, nvidia-smi ~= 6GB, i.e. near this card's ceiling
    # (B4 with the same config OOMs).
    batch_size = 3
    seq_len = 256
    pack = True
    val_ratio = 0.1
    num_workers = 0

    # --- corpora root (per-source raw/processed land here) ---
    corpora_dir = ProjectConfig.project_saved_path / "datasets"

    # --- source ids ---
    # LCCC: silver/lccc (MIT), spoken-Chinese base, base split only, streamed
    # and cleaned on the fly.
    lccc_hf_id = "silver/lccc"
    # Muice: Moemu/Muice-Dataset (CC-BY-NC-4.0, research use, requires
    # attribution and is non-commercial), 3541 hand-written companion
    # dialogues; the style anchor, kept at 100%.
    muice_hf_id = "Moemu/Muice-Dataset"

    # --- per-source raw/processed paths ---
    lccc_raw = corpora_dir / "lccc_raw.jsonl"
    lccc_processed = corpora_dir / "lccc_processed.jsonl"
    muice_raw = corpora_dir / "muice_raw.jsonl"
    muice_processed = corpora_dir / "muice_processed.jsonl"

    # --- sampling / mix (companion-style ratio) ---
    # Pretrain: 600k cleaned LCCC rows joined into {"text": ...} to teach
    # Chinese rhythm.
    # The old 70k cap was too tight: after cleaning it yielded only ~2.55M
    # tokens, and 20k steps x bs3 x seq256 = 15.4M tokens meant 6 epochs
    # over it, so the model started memorising the corpus.
    # Raw LCCC has 6.82M sessions (~375M tokens); relaxing the cap to 600k
    # rows (~22M tokens) keeps 20k steps under one epoch and still leaves
    # room to add more steps.
    lccc_pretrain_samples = 600000
    lccc_min_chars = 8
    # Relaxed cap: the old 512 dropped nearly every long dialogue, since
    # LCCC sessions average 75 chars but have a long tail.
    lccc_max_chars = 2048
    # SFT: muice upsampled x3 to set the style (~10k effective rows),
    # lccc contributes 5k rows to teach short replies.
    muice_upsample = 3.0
    lccc_sft_samples = 5000
    # (source, weight): the pipeline mixes by weight into the merged file.
    sft_mix = (("muice", 3.0), ("lccc_sft", 1.0))

    # --- merged outputs (read directly by train.py / eval.py) ---
    pretrain_raw_saved_path = corpora_dir / "lccc_raw.jsonl"
    pretrain_processed_saved_path = corpora_dir / "lccc_pretrain.jsonl"
    finetune_raw_saved_path = corpora_dir / "finetune_merged_raw.jsonl"
    finetune_processed_saved_path = corpora_dir / "finetune_merged.jsonl"

    # --- token cache (memory safe) ---
    # PretrainDataset keeps two copies of the corpus in RAM (a token list per
    # doc, plus the sliced windows). At 22M tokens that is ~2GB of Python ints,
    # which gets the process silently SIGKILLed by the OOM killer on a 14GB
    # box (no error message at all).
    # Instead the corpus is pre-tokenized into a single int32 numpy array
    # (22M tokens = 90MB) and windows are cut from it at train time via mmap.
    token_cache_dir = ProjectConfig.project_saved_path / "cache"
    pretrain_token_cache = token_cache_dir / "pretrain_tokens.npy"
    # Corpus the cache was built from: rebuild if the corpus mtime is newer,
    # otherwise we would train on stale text.
    token_cache_corpus = pretrain_processed_saved_path
    token_cache_max_chars = 2048
    

class ModelConfig:
    # --- model backbone (one shared decoder layer, looped) ---
    # Tall-and-narrow shape: hidden dimension halved, FFN widened.
    # Attention params scale as ~4d^2 (quadratic) while FFN scales as
    # ~3*d*ff (linear), so shrinking d and raising ff moves budget from
    # attention into the MLP and also halves the embedding (64000*d).
    #   d1536/ff24576/h12 => 230.5M total params (vs 323.8M for
    #   d3072/ff9216, -29%), measured peak 4.42GB, 0.52s/step.
    # Shape comparison (same seed, same data, pretrain 1000 steps):
    #   d1536/ff24576 -> ppl 2463.7 (faster early, overtakes by step 500)
    #   d3072/ff9216  -> ppl 2227.2 (stronger later, embedding 196M vs 98M)
    # Narrow shape chosen: 29% fewer params and faster early convergence;
    # the later gap still needs a long run to confirm.
    d_model = 1536
    n_heads = 12
    d_ff = 24576
    max_loop_times = 16
    lora_reusage = 4
    ranks = (8, 16, 32, 64)
    dropout = 0.0
    tie_embeddings = True

class TokenizerConfig:
    # Yi-1.5-64K tokenizer (vocab 64000) plus a local cache directory.
    # Versus Qwen2.5 (151665) the vocabulary is 58% smaller, dropping the
    # embedding from 310M to ~130M params; the freed memory widens the
    # Transformer trunk. The files are cached in tokenizer_dir and read
    # offline with from_pretrained.
    hf_id = "LLM-Research/Yi-1.5-6B"
    tokenizer_dir = ProjectConfig.project_saved_path / "tokenizer" / "yi-1.5-64k"
    trust_remote_code: bool = False
    label_ignore_token: int = -100

class TrainingConfig:
    # Single route: SGD(lr=1e-2, momentum=0.9) + whole model in bf16.
    max_steps = 20000
    learning_rate = 1e-2
    weight_decay = 0.01
    grad_clip = 1.0
    # lr schedule: linear warmup (actually min(warmup_steps, max_steps//10))
    # then cosine annealing down to min_lr_ratio * lr, so short runs are not
    # swallowed by the warmup.
    warmup_steps = 200
    min_lr_ratio = 0.05
    # fp32 master weight copy: compute in bf16, but the optimizer updates the
    # fp32 copy which is then written back to the bf16 model.
    # Measured necessity: bf16 resolves only ~0.004 around 1.0, which is
    # coarser than a single update, so updates get rounded away (this is
    # exactly why the residual gains used to stay pinned at 1.0). With the
    # copy on, the gains decay normally from 1.0 to 0.16.
    # Cost: one extra fp32 copy of the weights (~50% of the parameter count).
    fp32_master = True

    model_saved_dir = ProjectConfig.project_saved_path / "models_pretrain_v3"
    # Resume weights (optional): a .pt loaded before training. The SFT stage
    # points this at the pretrain best.pt for a pretrain -> finetune split.
    # Empty = random init.
    # Finished pretrain weights: saved/models_pretrain/best.pt (old Qwen
    # tokenizer version).
    init_from = ""
    eval_interval = 500


class ServeConfig:
    """Web usage layer (Streamlit): paths and runtime parameters.

    An empty checkpoint path means "auto-discover the newest checkpoint
    under saved/ (by mtime)"; an explicit path takes precedence. The
    sidebar can also override it at runtime.
    """

    # --- weights ---
    # Empty = scan saved/*/best.pt and take the newest usable one.
    checkpoint_path: str = ""
    # Whether auto-discovery may fall back to latest.pt.
    allow_latest_fallback: bool = True
    # Root directory scanned for weights (used by the sidebar).
    saved_root = ProjectConfig.project_saved_path

    # --- tokenizer ---
    # The checkpoint and tokenizer must match: this repo has weights saved
    # with both Yi-1.5-64K and Qwen2.5, whose vocabs are 64000 vs 151665.
    # A mismatch crashes outright, so the tokenizer directory is selected by
    # the checkpoint's vocab size.
    tokenizer_dir = ProjectConfig.project_saved_path / "tokenizer" / "yi-1.5-64k"
    tokenizer_dir_qwen = ProjectConfig.project_saved_path / "tokenizer" / "qwen2.5-0.5B"

    # --- dialogue ---
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    max_context_chars: int = 4000
    # Max length of a training sample (aligned with DataConfig.seq_len).
    max_length: int = 256
    # Separator between dialogue turns in the prompt. The SFT corpus is a
    # bare concatenation (no separator), so this defaults to empty to match
    # the training format exactly; changing it requires retraining the SFT.
    turn_separator: str = ""

    # --- feedback storage ---
    feedback_dir = ProjectConfig.project_saved_path / "feedback"
    feedback_file = feedback_dir / "edits.jsonl"
    # Weights produced by online learning (never overwrite the original
    # checkpoint).
    learned_model_dir = ProjectConfig.project_saved_path / "learned"


class FeedbackConfig:
    """Online learning (user edit -> backprop) hyperparameters.

    Precision route: bf16 weights + fp32 AdamW state + fp32 master copy.
    Static memory is about 0.86GB (backbone 57.5M params with the embedding
    frozen).
    """

    # --- architecture (must match the checkpoint) ---
    lora_reusage: int = 4
    max_loop_times: int = 16

    # --- optimizer ---
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # fp32 master copy: guarantees that tiny single-step updates are not
    # eaten by bf16 rounding.
    use_master: bool = True
    # Freeze the tied embedding. The embedding shares weights with the output
    # head, so its gradient is dense over the whole vocabulary (the bf16
    # gradient of 151665 x d2048 alone is 594MB, and with an fp32 master
    # copy plus AdamW state about 2.9GB), and changing it perturbs the
    # output distribution of every token at once. The signal from an online
    # edit should land in the backbone.
    freeze_embedding: bool = True

    # --- batch construction ---
    # Each learning round uses one batch: n_feedback user edits + n_anchor
    # anchor rows.
    n_feedback: int = 4
    n_anchor: int = 8
    # Anchor loss weight (regulariser, prevents catastrophic forgetting).
    anchor_weight: float = 1.0
    # Feedback loss weight.
    feedback_weight: float = 1.0
    # Anchor corpus source (prompt/response rows); empty disables anchors.
    anchor_corpus: str = str(
        ProjectConfig.project_saved_path / "datasets" / "finetune_merged.jsonl")
    # Replay buffer: already consumed feedback rows are mixed back into
    # training to prevent forgetting.
    replay_ratio: float = 0.3
    replay_capacity: int = 256

    # --- training ---
    # Epochs per "learning" round (the same batch is fitted repeatedly).
    epochs: int = 3
    # Filename for the weights saved after learning (under learned_model_dir).
    output_name: str = "learned.pt"
    # Whether to run a sanity check after training (compute loss on the
    # anchor set).
    validate_after: bool = True
