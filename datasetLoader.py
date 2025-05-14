from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from modules.util import resample_audio, get_muse_embeddings, clean_text_en, clean_text_ur
from modules.HypParams import *
import pandas as pd
import os
from g2p import englishG2P, urduG2P

CSV_PATH = os.path.join(os.path.dirname(__file__), "data/")
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "data/files_")

class AudioTranscriptDataset(Dataset):
    def __init__(self, mode):
        self.df = pd.read_csv(f"{CSV_PATH}{mode}.csv")
        self.audio_dir = f"{AUDIO_PATH}{mode}"
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["path"])
        if self.mode == "ur":
            transcript = clean_text_ur(row["sentence"])
        else:
            transcript = clean_text_en(row["sentence"])

        # Load waveform
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform, sample_rate = resample_audio(waveform, sample_rate)

        # Spectrogram
        mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=SPEC_MEL_CHANNELS,
            n_fft=SPEC_FRAME_SIZE,
            hop_length=SPEC_HOP_LENGTH,
            f_min=SPEC_FREQ_MIN,
            f_max=SPEC_FREQ_MAX
        )

        spec = mel_spec(waveform.mean(dim=0, keepdim=True))
        spec = (spec - spec.mean()) / spec.std()

        # Embeddings
        muse_embeddings, _ = get_muse_embeddings(transcript)

        if self.mode == "ur":
            gtp_embeddings = urduG2P.get_phoneme_embeddings(transcript)
            gtp_embeddings.insert(0, urduG2P.get_special_token_embedding("<sos>"))
            gtp_embeddings.append(urduG2P.get_special_token_embedding("<eos>"))
        else:
            gtp_embeddings = englishG2P.get_phoneme_embeddings(transcript)
            gtp_embeddings.insert(0, englishG2P.get_special_token_embedding("<s>"))
            gtp_embeddings.append(englishG2P.get_special_token_embedding("</s>"))

        return spec, muse_embeddings, gtp_embeddings