from torch import nn
from .specAugment import SpecAugment
from .convSubsampling import ConvSubsampling
from .conformerBlock import ConformerBlock
from .HypParams import *


class Conformer(nn.Module):
    def __init__(
        self,
        dim = CONFORMER_DIM,
        depth = CONFORMER_DEPTH,
        *,
        dim_head = CONFORMER_DIM_HEAD,
        heads = CONFORMER_HEADS,
        ff_mult = CONFORMER_FEED_FORWARD_EXPANSION_FACTOR,
        conv_expansion_factor = CONFORMER_CONV_EXPANSION_FACTOR,
        conv_kernel_size = CONFORMER_CONV_KERNEL_SIZE,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal

            ))

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x
    
class FullConformer(nn.Module):
    def __init__(self, *, dim, depth):
        super().__init__()
        self.spec_augment = SpecAugment()
        self.subsampling = ConvSubsampling(in_channels=1, out_channels=dim)
        self.linear = nn.Linear(dim * ((SPEC_MEL_CHANNELS-1)//4), dim)
        self.dropout = nn.Dropout(0.1)
        self.conformer = Conformer(dim, depth)


    def forward(self, x):
        x = self.spec_augment(x)  
        x = self.subsampling(x)
        x = self.linear(x)
        x = self.dropout(x)       
        x = self.conformer(x)
        return x