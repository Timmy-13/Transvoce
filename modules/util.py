import re
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch
from .HypParams import *


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def resample_audio(waveform, sample_rate):
    if sample_rate != SPEC_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = SPEC_SAMPLE_RATE
    return waveform, sample_rate



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

def clean_text_en(text):

    text = text.replace("’", "'")

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove all characters except letters, whitespace, and apostrophes
    text = re.sub(r"[^a-zA-Z\s']", '', text)

    # Remove apostrophes not between two letters
    text = re.sub(r"(?<![a-zA-Z])'(?![a-zA-Z])", '', text)  # lone apostrophes
    text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '', text)  # leading or trailing

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_ur(text):

    text = text.replace("’", "'")

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove all characters except letters, whitespace, and apostrophes
    text = re.sub(r"[^a-zA-Z\s']", '', text)

    # Remove apostrophes not between two letters
    text = re.sub(r"(?<![a-zA-Z])'(?![a-zA-Z])", '', text)  # lone apostrophes
    text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '', text)  # leading or trailing

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Dummy MUSE embeddings (to be replaced with real ones later)
def get_muse_embeddings(transcript):
    words = transcript.strip().split()
    n = len(words)
    return torch.zeros(n, 300), n  # (n_words, 300-dim)
