from torch import Tensor, nn

from carina.model.attention import CarinaMultiHeadAttention
from carina.model.kv_cache import CarinaKVCache


class CarinaDecoder(nn.Module):
    def __init__(
        self,
        kv_cache: CarinaKVCache,
        attn: CarinaMultiHeadAttention,
        mlp: nn.Module,
        rope: nn.Module,
    ) -> None:
        super().__init__()
        self.kv_cache = kv_cache
        self.attn = attn
        self.mlp = mlp
        self.rope = rope
        self.norm_attn = nn.LayerNorm(attn.d_model)
        self.norm_mlp = nn.LayerNorm(attn.d_model)

    def forward(self, x: Tensor, cache: CarinaKVCache | None = None) -> Tensor:
        x = x + self.attn(self.norm_attn(x), self.rope, cache)
        x = x + self.mlp(self.norm_mlp(x))
        return x
