"""Causal wrapper: embed + looped backbone + head, with prefill/generate."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from carina.model.carina_model import CarinaModel
from carina.model.kv_cache import CarinaKVCache

from .checks import (
    check_id_batch,
    check_is,
    check_loss_pair,
    check_not_none,
    check_pos_int,
    check_single_token,
    check_tensor,
)


class CarinaCausalModel(nn.Module):
    """Full causal LM: ids -> looped hidden -> logits.

    Args:
        vocab_size: Vocabulary size (must be > 0).
        d_model: Model width.
        d_hidden: MLP hidden width.
        n_heads: Attention heads.
        max_loop_times: Backbone loop count.
        lora_reusage: Iterations sharing one complement group.
        ranks: One rank per complement group (len == max_loop_times // lora_reusage).
        tie: If True, output head shares the embedding weight.
        ignore_index: Label value ignored by the loss.
    """

    vocab_size: int
    d_model: int
    ignore_index: int
    max_seq_len: int

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_hidden: int = 2048,
        n_heads: int = 8,
        max_loop_times: int = 16,
        lora_reusage: int = 4,
        ranks: tuple[int, ...] = (8, 16, 32, 64),
        alpha: float = 16.0,
        dropout: float = 0.2,
        max_seq_len: int = 2048,
        tie: bool = True,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        check_pos_int("Causal.init", "vocab_size", vocab_size)
        check_is("Causal.init", "ignore_index", ignore_index, int)
        self.vocab_size: int = vocab_size
        self.d_model: int = d_model
        self.ignore_index: int = ignore_index
        self.max_seq_len: int = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.backbone = CarinaModel(
            max_loop_times=max_loop_times,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            lora_reusage=lora_reusage,
            ranks=ranks,
            alpha=alpha,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.tie: bool = tie
        self.lm_head: nn.Linear | None = None
        if not tie:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _logits_from_hidden(self, hidden_states: Tensor) -> Tensor:
        """Apply final norm + head to hidden [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]."""
        normed_states: Tensor = self.final_norm(hidden_states)
        if self.tie:
            return normed_states @ self.embed.weight.T
        check_not_none("Causal.head", "lm_head", self.lm_head)
        return self.lm_head(normed_states)

    def _check_ids(self, input_ids: Tensor) -> None:
        """Validate [batch_size, seq_len] ids are in [0, vocab_size)."""
        check_id_batch("Causal.ids", input_ids, self.vocab_size)

    def forward(
        self, input_ids: Tensor, caches: list[CarinaKVCache] | None = None
    ) -> Tensor:
        """Map ids [batch_size, seq_len] to logits [batch_size, seq_len, vocab_size].

        Args:
            input_ids: Token ids [batch_size, seq_len].
            caches: One cache per loop iteration, or None for pure
                full forward (training). Non-None means prefill/decode.
        """
        self._check_ids(input_ids)
        embedded_states: Tensor = self.embed(input_ids)
        backbone_states: Tensor = self.backbone(embedded_states, caches)
        return self._logits_from_hidden(backbone_states)

    def prefill(
        self, input_ids: Tensor
    ) -> tuple[Tensor, list[CarinaKVCache]]:
        """Prefill prompt: fill per-iteration caches, return logits + caches.

        Args:
            input_ids: Prompt ids [batch_size, seq_len].

        Returns:
            Tuple (logits [batch_size, seq_len, vocab_size], caches ready for decode_step).
        """
        self._check_ids(input_ids)
        truncated_ids: Tensor = input_ids[:, -self.max_seq_len:]
        caches: list[CarinaKVCache] = self.backbone.new_caches()
        logits: Tensor = self.forward(truncated_ids, caches)
        return logits, caches

    def decode_step(
        self, token_ids: Tensor, caches: list[CarinaKVCache]
    ) -> Tensor:
        """Decode one step with caches: [batch_size, 1] -> logits [batch_size, 1, vocab_size]."""
        check_tensor("Causal.decode", "token_ids", token_ids, 2)
        check_single_token("Causal.decode", token_ids)
        check_is("Causal.decode", "caches", caches, list)
        self._check_ids(token_ids)
        embedded_states: Tensor = self.embed(token_ids)
        backbone_states: Tensor = self.backbone(embedded_states, caches)
        return self._logits_from_hidden(backbone_states)

    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy over [batch_size, seq_len, vocab_size] vs [batch_size, seq_len] with ignore_index."""
        check_loss_pair("Causal.loss", logits, labels)
        return F.cross_entropy(
            logits.float().reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

    @torch.no_grad()
    def generate(
        self, input_ids: Tensor, max_new_tokens: int,
        eos_id: int | None = None, use_cache: bool = True,
    ) -> Tensor:
        """Greedy decode, optionally with KV-cache prefill + decode steps.

        Args:
            input_ids: Prompt ids [batch_size, seq_len].
            max_new_tokens: Tokens to generate (must be > 0).
            eos_id: Stop when all batches emit it (or None to ignore).
            use_cache: True = prefill once + cached decode steps;
                False = full forward each step (slow, for correctness check).
        """
        check_tensor("Causal.generate", "input_ids", input_ids, 2)
        check_pos_int("Causal.generate", "max_new_tokens", max_new_tokens)
        was_training: bool = self.training
        self.eval()
        generated_ids: Tensor = input_ids
        if use_cache:
            logits, caches = self.prefill(generated_ids)
            for _ in range(max_new_tokens):
                next_token_ids: Tensor = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_ids.to(generated_ids.device)], dim=1)
                if eos_id is not None and bool((next_token_ids == eos_id).all()):
                    break
                logits = self.decode_step(next_token_ids, caches)
        else:
            for _ in range(max_new_tokens):
                truncated_ids: Tensor = generated_ids[:, -self.max_seq_len:]
                logits = self.forward(truncated_ids)
                next_token_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_ids.to(generated_ids.device)], dim=1)
                if eos_id is not None and bool((next_token_ids == eos_id).all()):
                    break
        if was_training:
            self.train()
        return generated_ids
