import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_960H
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load Pre-trained Wav2Vec2 Model
bundle = WAV2VEC2_ASR_LARGE_960H
model = bundle.get_model()

# Load the vocabulary (characters/phonemes from Wav2Vec2)
vocab = bundle.get_labels()

# Extended ARPAbet phoneme vocabulary
phoneme_vocab = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def preprocess_audio(waveform, sample_rate, target_sample_rate=16000):
    if sample_rate != target_sample_rate:
        waveform = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform

def phoneme_recognition(waveform, model):
    model.eval()
    waveform = waveform.squeeze(0).unsqueeze(0)  # Ensure correct shape [1, T]

    with torch.no_grad():
        emissions, _ = model(waveform)  # Output shape: [1, time_steps, vocab_size]

    return emissions.squeeze(0)  # Remove batch dim → [time_steps, vocab_size]

def decode_emissions(emissions, vocab):
    predicted_ids = emissions.argmax(dim=-1)  # Get max prob index at each time step
    predicted_tokens = [vocab[idx.item()] for idx in predicted_ids]

    # Remove consecutive duplicates and blank tokens ("_")
    phonemes = []
    prev = None
    for token in predicted_tokens:
        if token != prev and token != "_":
            phonemes.append(token)
        prev = token

    return phonemes

# Load and process audio
audio_file = 'path_to_audio.wav'
waveform, sample_rate = load_audio(audio_file)
waveform = preprocess_audio(waveform, sample_rate)

print(waveform.shape)
# Predict and decode phonemes
emissions = phoneme_recognition(waveform, model)
phonemes = decode_emissions(emissions, vocab)
print(len(phonemes))
print("Predicted Phonemes:", phonemes)
