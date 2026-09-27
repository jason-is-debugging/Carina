"""Top-level Carina transformer body and language-model heads.

Two classes are exposed:

* :class:`CarinaModel` — the transformer body without an LM head.
* :class:`CarinaForCausalLM` — the body plus a (potentially tied)
  LM head and a tiny loss helper.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import CarinaConfig
from .embedding import CarinaEmbedding
from .layers import CarinaDecoderLayer
from .rmsnorm import CarinaRMSNorm


class CarinaModel(nn.Module):
    """Carina transformer body (no LM head).

    Args:
        config: A :class:`carina.config.CarinaConfig` instance.
    """

    def __init__(self, config: CarinaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = CarinaEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [CarinaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CarinaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        """Run the transformer body.

        Returns ``(hidden_states, presents)`` where ``presents`` is a
        per-layer list of ``(k, v)`` tensors (or ``None`` when
        ``use_cache`` is False).
        """
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        presents: list | None = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[idx]
            hidden_states, new_past = layer(
                hidden_states,
                past_key_value=past,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            if presents is not None:
                presents.append(new_past)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class CarinaForCausalLM(nn.Module):
    """Carina language model: body + (tied) LM head.

    Args:
        config: A :class:`carina.config.CarinaConfig` instance.
    """

    def __init__(self, config: CarinaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = CarinaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def post_init(self) -> None:
        """Apply standard small-init scaling to linear / embedding layers."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass returning logits + optional loss."""
        hidden_states, presents = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": presents,
            "hidden_states": hidden_states,
        }

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        """Return the total trainable parameter count."""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            n -= self.model.embed_tokens.weight.numel()
        return n

    def tie_weights(self) -> None:
        """Re-tie the LM head to the embedding matrix (idempotent)."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


__all__ = ["CarinaForCausalLM", "CarinaModel"]
