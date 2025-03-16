from torch import nn
import torchaudio.transforms as T

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