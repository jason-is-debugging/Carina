# Experiment Handoff

State of the project as of the squashed commit, plus everything that was
measured. Written so a fresh agent can pick up the work without needing
the conversation history.

## 1. What the model is

One shared decoder layer (causal attention + SwiGLU MLP) looped `K`
times per forward pass, with per-group low-rank depth complements that
specialise the shared MLP for different rounds.

```
embedding -> [attention -> MLP] x K -> output projection (tied)
                 ^                   ^
                 |                   +-- per-round depth complement (LoRA)
                 +-- per-round KV cache, per-round residual gain
```

* Depth complement groups: `group = round_index // lora_reusage`. With
  `lora_reusage=4` and `K=16` there are 4 groups (ranks 8/16/32/64).
* `stream_norm` (LayerNorm after each round's residual add) is required:
  without it the hidden norm explodes across the loop (33 -> 980).
* `attn_gain` / `mlp_gain`: one learnable scalar per round.

## 2. Environment

| | |
|---|---|
| Local box | RTX 4060 Laptop, 8GB VRAM, 14.6GB RAM |
| Remote box | RTX 4090, 24GB VRAM (rented, not yet connected) |
| Python | 3.11, deps via `uv` (`uv sync`) |
| GPU peak | 4.05GB at K=4, 6.20GB at K=16 |

## 3. Measured results

### 3.1 K sweep, trained from scratch

Identical seed, identical data, identical step count (4000 steps =
~3.1M tokens seen). `lora_reusage=4`.

| K | depth-complement groups | params | val_ppl | time | peak |
|---|---|---|---|---|---|
| 4 | 1 (rank 8, shared by all 4 rounds) | 221.7M | **810.87** | 14.3 min | 4.05GB |
| 8 | 2 (ranks 8, 16) | 222.9M | 851.15 | 22.4 min | 4.75GB |
| 16 | 4 (ranks 8, 16, 32, 64) | 230.5M | (running) | - | 6.20GB |

Per-round gains after training:

```
K=4   attn_gain = [0.245, 0.100, 0.099, 0.098]
K=8   attn_gain = [0.217, 0.105, 0.104, 0.104, 0.103, 0.102, 0.100, 0.098]
K=16  attn_gain = [0.363, 0.443, 0.443, 0.443, ... 0.443, ..., 0.424]
```

Two things to read out of this:

1. **The model compresses its own effective depth.** After 4000 steps the
   K=4 model puts 0.245 on round 1 and ~0.10 on every later round, i.e.
   rounds 2-4 are nearly identical. After only 1500 steps the same
   configuration showed `[0.258, 0.393, 0.389, 0.385]` - the collapse
   happens *during* training, not at init.
2. **At K=16 the gain signature degenerates**: rounds 2-10 end up with
   *exactly* the same value. The shared loop is too deep for the weights
   to differentiate its rounds at all.

### 3.2 Why K=4 wins despite being the least expressive

This is the confound to be careful about. `lora_reusage=4` means:

| K | groups | ranks |
|---|---|---|
| 4 | **1** | (8,) |
| 8 | 2 | (8, 16) |
| 16 | 4 | (8, 16, 32, 64) |

At K=4 **all four rounds share a single rank-8 complement**, so the
rounds have no dedicated parameters to differentiate themselves with.
K=8 and K=16 have 2 and 4 groups respectively. So "K=4 wins" partly
means "one group is easier to train than four", not "depth does not pay".

### 3.3 Parameter budget (d=1536, ff=24576, vocab=64000)

Shared layer = 122.8M params.

| configuration | depth complements | % of shared | total |
|---|---|---|---|
| K=4, 1 group (current) | 0.63M | 0.51% | 221.7M |
| K=4, one per round | 9.40M | 7.66% | 230.5M |
| K=8, one per round | 29.45M | 23.99% | 250.5M |
| K=16, one per round | 69.56M | 56.67% | 290.6M |

Stacked baseline at the same budget (full-rank distinct layers):

| stacked layers | params | fits the 230.5M budget? |
|---|---|---|
| L=1 | 221.1M | yes |
| L=2 | 343.8M | no (+49%) |
| L=4 | 589.3M | no (+156%) |

**This is the core argument for looping**: making 4 rounds differentiable
costs 122.8M as a stacked model but only 9.4M as low-rank complements -
a 13x difference. The reason looping is worth anything is `O(K*r*d)`
instead of `O(K*d^2)`.

The caveat: at K=16 with a complement per round the depth parameters
reach 56.67% of the shared layer, so the loop has degenerated back into
something stacked. There is an upper bound on useful K.

### 3.4 Cheap ways to differentiate rounds

| mechanism | cost at K=16 | % of shared layer |
|---|---|---|
| per-round LoRA on gate/up/down | 69.6M | 56.67% |
| **round embedding + per-round gain + per-round LN affine** | **0.074M** | **0.06%** |
| per-round bias / scale | ~0 | ~0 |

The middle row is essentially free and is probably the better mechanism:
give each round a `d`-dimensional identity vector added to the hidden
state, so the shared weights can condition on "which round am I". Note
the current depth complements only touch the MLP; **attention is fully
shared**, which is the likely reason later rounds collapse onto each
other.

## 4. Fixes already made (do not regress these)

1. **bf16 rounding ate the parameter updates.** bf16 resolves only
   ~0.004 around 1.0, coarser than a single update, so the residual
   gains sat at exactly 1.0. Fixed by an fp32 master weight copy in
   `Trainer` (`TrainerOptions.fp32_master`, default on). Compute is
   still bf16.
2. **Residual stream norm explosion.** Fixed by `stream_norm`.
3. **OOM killer, silently.** `PretrainDataset` kept two copies of the
   corpus as Python int lists (~2GB at 22M tokens); the process was
   SIGKILLed with no error message on a 14GB box. Fixed by the token
   cache (`src/carina/datasets/token_cache.py`), which stores one int32
   numpy array (90MB) and serves mmap'd windows.
4. **Tokenizer.** Yi-1.5-64K (vocab 64000), cached locally in
   `saved/tokenizer/yi-1.5-64k/`. Qwen2.5 was the previous choice
   (vocab 151665) and its embedding was 310M of 368M params.

## 5. Open questions, in priority order

### 5.1 Is there a stacked baseline in this repo?

**No.** A param-matched stacked baseline existed at commit `37aa0ca`
(`src/carina/depth_model.py`) and was deleted in `7f78bda` during a
package restructure. Recover it with:

```bash
git show 37aa0ca:src/carina/depth_model.py
```

Without it, "is looping worth it" has never been answered inside this
project. This is the highest-value missing experiment.

### 5.2 Fair K=4 comparison

The current K=4 has only 1 complement group, so it cannot differentiate
its rounds. Compare `lora_reusage=1` (one complement per round, ranks
8/16/32/64) against the current `lora_reusage=4` at the same K. If the
per-round version wins, the "depth needs dedicated parameters"
hypothesis holds.

### 5.3 Round identity instead of per-round matrices

Add a learnable round embedding (K x d) added to the hidden state at the
start of each round, plus per-round LayerNorm affine. Cost is ~0.06% of
the trunk at K=16. Test whether this lets the rounds differentiate
attention (which no current mechanism touches).

### 5.4 Long training

The user's stated criterion is "same data volume trained, time does not
matter". All numbers above are 4000 steps (~3.1M tokens, 0.14 epochs of
the 22.5M corpus). A serious run should be 40k+ steps. On the 4090 this
is very affordable; on the 4060 it is not.

### 5.5 Shape re-check

A shape comparison at 1000 steps gave d1536/ff24576 -> ppl 2463.7 and
d3072/ff9216 -> ppl 2227.2. The wide shape was stronger later because of
its larger embedding (196M vs 98M). The narrow shape was kept because it
converged faster early and uses 29% fewer parameters. This was never
re-checked at longer horizons, and it matters more now that the
embedding dominates the parameter budget.

### 5.6 Generation quality

Greedy decoding degenerated into a newline loop. Root causes found:
LCCC is multi-turn dialogue joined by newlines, so predicting `\n` is the
unigram-dominant transition; and the pretrain corpus was too small (2.55M
tokens, 6 epochs). Both are now addressed (22.5M tokens). **Generation
quality has not been re-checked since.** Do this before drawing any
conclusion about model quality from perplexity alone.

## 6. Data locations

```
saved/datasets/lccc_raw.jsonl         6.82M raw sessions (~375M tokens)
saved/datasets/lccc_pretrain.jsonl    600k rows, 22.5M tokens
saved/datasets/finetune_merged.jsonl  15,149 rows (Muice x3 + LCCC 5k)
saved/cache/pretrain_tokens.npy       22.5M tokens, int32, 90MB
saved/tokenizer/yi-1.5-64k/           local tokenizer cache
```

The cache is rebuilt automatically when the corpus mtime is newer than
the cache mtime.

## 7. How to run

```bash
uv sync
uv run python scripts/prepare_data.py     # build the corpora
uv run python scripts/train.py             # pretrain (config-driven)
uv run python scripts/train.py --mode finetune
```

`scripts/train.py` deliberately keeps only `--mode` and `--max-steps`.
Everything else is configured in `src/carina/config.py`; the user asked
explicitly not to add more CLI arguments.

## 8. Config quick reference

`src/carina/config.py`:

* `ModelConfig`: d_model=1536, n_heads=12, d_ff=24576, max_loop_times=16,
  lora_reusage=4, ranks=(8,16,32,64), dropout=0.0, tie_embeddings=True
* `DataConfig`: batch_size=3, seq_len=256, pack=True, val_ratio=0.1
* `TrainingConfig`: max_steps=20000, learning_rate=1e-2, weight_decay=0.01,
  grad_clip=1.0, warmup_steps=200, min_lr_ratio=0.05, fp32_master=True
* `TokenizerConfig`: Yi-1.5-64K, vocab 64000
