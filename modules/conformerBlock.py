from torch import nn
from .feedForward import FeedForward
from .attention import Attention
from .conformerConvModule import ConformerConvModule
from .util import Scale, PreNorm
from .HypParams import *



class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads,
        ff_mult,
        conv_expansion_factor,
        conv_kernel_size,
        attn_dropout = CONFORMER_BLOCK_ATTN_DROPOUT,
        ff_dropout = CONFORMER_BLOCK_FF_DROPOUT,
        conv_dropout = CONFORMER_BLOCK_CONV_DROPOUT,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x