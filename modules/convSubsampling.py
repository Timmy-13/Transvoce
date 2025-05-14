from torch import nn

class ConvSubsampling(nn.Module):
    def __init__(self, out_channels, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)  # Shape: (B, C, H, W) -> (B, C, H/4, W/4)
        b, c, h, w = x.shape
        x = x.permute(0,3,1,2)
        x = x.contiguous().view(b, w, c*h)  # Reshape for Transformer input
        return x