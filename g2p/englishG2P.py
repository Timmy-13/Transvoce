import torch
import os
from g2p_en import G2p

EMBEDDING_DIM = 512
MODEL_NAME = "resources/phoneme_embedding_en.pt"
EN_EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g2p = G2p()

G2P_PHONEMES_LIST = g2p.phonemes
G2P_PHONEMES_DICT = {p: i for i, p in enumerate(G2P_PHONEMES_LIST)}
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>"]

# Load pre-trained embeddings
embedding = torch.nn.Embedding(num_embeddings=len(G2P_PHONEMES_LIST), embedding_dim=EMBEDDING_DIM).to(device)
embedding.load_state_dict(torch.load(EN_EMBEDDING_PATH))

def g2p_text(text):
    """
    Converts text to a list of phonemes.
    """
    phonemes = g2p(text)
    return [p if p != " " else "<pad>" for p in phonemes]

def get_phoneme_embeddings(sentence):
    """
    Converts sentence to phoneme embeddings.
    """
    phoneme_embeddings = []
    phonemes = g2p_text(sentence)
    for p in phonemes:
        if p in G2P_PHONEMES_DICT:
            phoneme_idx = torch.tensor([G2P_PHONEMES_DICT[p]]).to(device)  # Ensure tensor is on GPU
            phoneme_embeddings.append(embedding(phoneme_idx).detach())
        else:
            print(f"Warning: Phoneme '{p}' not found in dictionary.")
            phoneme_embeddings.append(torch.zeros(EMBEDDING_DIM))  # Fallback to zero embedding
    return phoneme_embeddings

def get_special_token_embedding(token):
    """
    Returns the embedding for special tokens.
    """
    if token in SPECIAL_TOKENS:
        phoneme_idx = torch.tensor([G2P_PHONEMES_DICT[token]]).to(device)  # Ensure tensor is on GPU
        return embedding(phoneme_idx).detach()
    else:
        print("WARN: NOT A SPECIAL TOKEN")
        return get_phoneme_embeddings(token)  # Fallback to G2P conversion for unknown tokens

def g2p_tokenwise(sentence):
    """
    Converts sentence to tokenwise phoneme lists.
    """
    phonemes = []
    for word in sentence.split(" "):
        phonemes.append(g2p_text(word))
    return phonemes
