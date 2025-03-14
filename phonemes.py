import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.models import wav2vec2
from torchaudio.pipelines import WAV2VEC2_LARGE
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Step 1: Load Pre-trained Wav2Vec2 Model (or DeepSpeech)
model = WAV2VEC2_LARGE.get_model()

# Step 2: Load and preprocess the audio file
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# Preprocess the audio (convert to the appropriate sample rate for the model)
def preprocess_audio(waveform, sample_rate, target_sample_rate=16000):
    if sample_rate != target_sample_rate:
        resample_transform = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
    return waveform

# Step 3: Pass through the model to get the prediction
def phoneme_recognition(waveform, model):
    # Ensure model is in eval mode
    model.eval()

    # Squeeze the waveform to remove the extra dimension (if needed)
    waveform = waveform.squeeze(0)  # Remove the batch dimension if it's 1

    # Ensure the waveform is now a 2D tensor with shape (1, time_steps)
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # Add batch dimension back

    with torch.no_grad():
        # Forward pass through the model
        emissions, _ = model(waveform)
    
    # We will now decode the model's output to phonemes or words (depending on your setup)
    return emissions

# Assuming you have a list of phonemes or words for the model
phoneme_vocab = ["aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "el", "en", "er", "ey", "f", "g", "h", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh", "t", "th", "uh", "uw", "v", "w", "y", "z", "zh"]

# Load an example audio file (adjust the file path)
audio_file = 'path_to_audio.wav'

# Step 4: Process and Predict
waveform, sample_rate = load_audio(audio_file)
waveform = preprocess_audio(waveform, sample_rate)

# Run phoneme prediction
emissions = phoneme_recognition(waveform, model)

print("Predicted Phonemes:", emissions)
