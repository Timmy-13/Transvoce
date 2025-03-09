import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import TimeMasking, FrequencyMasking
import torchaudio.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange
nn.TransformerDecoder
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 36,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads # 36 * 4
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)  # W0
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
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

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 32,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
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

# Conformer

class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 4,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
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

# SpecAugment Module
class SpecAugment(nn.Module):
    def __init__(self, freq_blocks=2, time_blocks=10, 
                 freq_block_ratio=0.33, time_block_ratio=0.05):
        super().__init__()
        self.freq_blocks = freq_blocks
        self.time_blocks = time_blocks
        self.freq_block_ratio = freq_block_ratio
        self.time_block_ratio = time_block_ratio

    def forward(self, x):
        _, _, freq_bins, time_steps = x.shape  # Get spectrogram shape

        # Apply frequency masking
        for _ in range(self.freq_blocks):
            max_length = int(freq_bins * self.freq_block_ratio)
            x = T.FrequencyMasking(max_length)(x)

        # Apply time masking
        for _ in range(self.time_blocks):
            max_length = int(time_steps * self.time_block_ratio)
            x = T.TimeMasking(max_length)(x)

        return x

# Convolutional Subsampling Module
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

class FullConformer(nn.Module):
    def __init__(self, *, dim, depth):
        super().__init__()
        self.spec_augment = SpecAugment()
        self.subsampling = ConvSubsampling(in_channels=1, out_channels=dim)

        self.linear = nn.Linear(dim*((128-1)//4), dim)
        self.dropout = nn.Dropout(0.1)
        
        self.conformer = Conformer(
            dim=dim,
            depth=depth,
            dim_head=36,
            heads=4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=32,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1,
            conv_causal=False
        )

    def forward(self, x):
        x = self.spec_augment(x)  
        x = self.subsampling(x)  
        print(x.shape)
        x = self.linear(x)
        print(x.shape)
        x = self.dropout(x)       
        x = self.conformer(x)     
        return x
    

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("src.wav")

# Ensure the sample rate is 16,000 Hz
if sample_rate != 16000:
    resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Mel spectrogram parameters
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=128,         # 128 Mel channels
    n_fft=int(0.050 * sample_rate),  # Frame size = 50ms
    hop_length=int(0.0125 * sample_rate),  # Frame step = 12.5ms
    f_min=20,           # Lower band = 20 Hz
    f_max=8000          # Upper band = 8000 Hz
)

# Convert to mono & compute Mel spectrogram
spec = mel_spectrogram(waveform.mean(dim=0, keepdim=True))
spec = (spec - spec.mean()) / spec.std()  # Normalize
spec = spec.unsqueeze(0)  # Add batch dimension

# Define and run model
model = FullConformer(dim=144, depth=16)
output = model(spec)

print(output.shape)  # Expected: (B, Time, num_classes)