import torch
from tqdm import tqdm
from modules.util import compute_muse_loss
from modules.HypParams import *
from modules.conformer import FullConformer
from datasetLoader import AudioTranscriptDataset
from torch.utils.data import DataLoader
from modules.attention import Attention
from modules.decoder import PhonemeDecoder
from transvoce_model import TransvoceModel

SAVE_EVERY = 10

def train_step(spec, muse_embeddings, gtp_embeddings, model, optimizer):

    model_output, phoneme_loss = model(spec, gtp_embeddings)  # uses wrapper forward

    muse_loss = compute_muse_loss(model_output, muse_embeddings)
    total_loss = muse_loss + phoneme_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
dataset = AudioTranscriptDataset("ur")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Component initialization
conformer = FullConformer(dim=CONFORMER_DIM, depth=CONFORMER_DEPTH)
attention = Attention(dim=512, dim_head=64, heads=8, dropout=0.2)
decoder = PhonemeDecoder()
enc_att_proj = torch.nn.Linear(300, 512)

# Wrapper model
wrapper_model = TransvoceModel(conformer, attention, decoder, enc_att_proj).to(device)
optimizer = torch.optim.Adam(wrapper_model.parameters(), lr=1e-4)



for epoch in range(1, 101):
    print(f"\nEpoch {epoch}")
    wrapper_model.train()
    epoch_loss = 0.0

    for batch_idx, (spec, muse_emb, gtp_embeddings) in enumerate(tqdm(loader, desc="Training")):
        try:
            spec = spec.to(device)

            loss = train_step(spec, muse_emb, gtp_embeddings, wrapper_model, optimizer)
            epoch_loss += loss
            # tqdm.write(f"Batch {batch_idx+1} Loss: {loss:.4f}")

        except Exception as e:
            tqdm.write(f"Error in batch {batch_idx+1}: {e}")

    print(f"Avg Loss: {epoch_loss / len(loader):.4f}")

    # Save model checkpoint
    if epoch % SAVE_EVERY == 0:
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': wrapper_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")