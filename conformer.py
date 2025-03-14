from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from SpecAugment import SpecAugment
from ConvSubsampling import ConvSubsampling
from ConformerBlock import ConformerBlock
from util import resample_audio
from HypParams import *

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
        self.linear = nn.Linear(dim*((SPEC_MEL_CHANNELS-1)//4), dim)
        self.dropout = nn.Dropout(0.1)
        self.conformer = Conformer(dim, depth)


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
waveform, sample_rate = resample_audio(waveform, sample_rate)
# Ensure the sample rate is 16,000 Hz

# Mel spectrogram parameters
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=SPEC_MEL_CHANNELS,         # 128 Mel channels
    n_fft=SPEC_FRAME_SIZE,  # Frame size = 50ms
    hop_length=SPEC_HOP_LENGTH,  # Frame step = 12.5ms
    f_min=SPEC_FREQ_MIN,           # Lower band = 20 Hz
    f_max=SPEC_FREQ_MAX          # Upper band = 8000 Hz
)

# Convert to mono & compute Mel spectrogram
spec = mel_spectrogram(waveform.mean(dim=0, keepdim=True))
spec = (spec - spec.mean()) / spec.std()  # Normalize
spec = spec.unsqueeze(0)  # Add batch dimension

# Define and run model
model = FullConformer(dim = CONFORMER_DIM, depth = CONFORMER_DEPTH)
output = model(spec)

print(output.shape)  # Expected: (B, Time, num_classes)