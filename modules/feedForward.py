from torch import nn
from .util import Swish

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), # 35*144 into 35*576
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim), #reverse
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)