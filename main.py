import torch
from torch.nn.functional import mse_loss
from modules.HypParams import *
from modules.conformer import FullConformer
from datasetLoader import AudioTranscriptDataset
from torch.utils.data import DataLoader
from modules.attention import Attention
from modules.decoder import PhonemeDecoder

# Compute MUSE loss
def compute_muse_loss(model_output, muse_embeddings):
    model_output = model_output.squeeze(0)  # (Time, D)
    muse_embeddings = muse_embeddings.to(model_output.device)
    muse_embeddings = muse_embeddings.squeeze(0)
    n = min(model_output.shape[0], muse_embeddings.shape[0])
    return mse_loss(model_output[:n], muse_embeddings[:n])  # Eqn from image

def train_step(spec, muse_embeddings, gtp_embeddings, model, optimizer, attention, decoder, enc_att_proj):
    device = next(model.parameters()).device
    spec = spec.to(device)
    muse_embeddings = muse_embeddings.to(device)
    model_output = model(spec)  # (1, T_audio, D)

    expanded_model_output = enc_att_proj(model_output)
    total_phoneme_loss = 0
    for gtp in gtp_embeddings:
        gtp = gtp.to(device)  # (1, 1, 300)

        # Attention: gtp is query, model_output is context
        attended = attention(gtp, context=expanded_model_output)  # (1, 1, D)

        # Decode attended to predict phoneme embedding
        pred = decoder(gtp, attended)  # (1, 1, 300)

        # MSE loss per phoneme
        loss = torch.nn.MSELoss()(pred.squeeze(0), gtp.squeeze(0))
        total_phoneme_loss += loss

    avg_phoneme_loss = total_phoneme_loss / len(gtp_embeddings)

    muse_loss = compute_muse_loss(model_output, muse_embeddings)

    total_loss = muse_loss + avg_phoneme_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()


dataset = AudioTranscriptDataset("ur")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = FullConformer(dim=CONFORMER_DIM, depth=CONFORMER_DEPTH)
device = next(model.parameters()).device
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
attention = Attention(dim = 512, dim_head = 64, heads = 8, dropout = 0.2)
decoder = PhonemeDecoder()
enc_att_proj = torch.nn.Linear(300, 512).to(device)

for epoch in range(100):
    print(f"\nEpoch {epoch+1}")
    for spec, muse_emb, gtp_embeddings in loader:
        try:
            loss = train_step(spec, muse_emb, gtp_embeddings, model, optimizer, attention, decoder, enc_att_proj)
            print(f"Loss: {loss:.4f}")
        except Exception as e:
            print(f"Error: {e}")
