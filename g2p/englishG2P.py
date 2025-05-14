import torch
import os
from g2p_en import G2p

EMBEDDING_DIM = 512
MODEL_NAME = "resources/phoneme_embedding_en.pt"
EN_EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)

g2p = G2p()

G2P_PHONEMES_LIST = g2p.phonemes
G2P_PHONEMES_DICT = {p: i for i, p in enumerate(G2P_PHONEMES_LIST)}
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>"]


embedding = torch.nn.Embedding(num_embeddings=len(G2P_PHONEMES_LIST), embedding_dim=EMBEDDING_DIM)
embedding.load_state_dict(torch.load(EN_EMBEDDING_PATH))

def g2p_text(text):
    phonemes = g2p(text)
    return [p if p != " " else "<pad>" for p in phonemes]


def get_phoneme_embeddings(sentence):
    phoneme_embeddings = []
    phonemes = g2p_text(sentence)
    for p in phonemes:
        phoneme_idx = torch.tensor([G2P_PHONEMES_DICT[p]])
        phoneme_embeddings.append(embedding(phoneme_idx).detach())
    return phoneme_embeddings

def get_special_token_embedding(token):
    if token in SPECIAL_TOKENS:
        phoneme_idx = torch.tensor([G2P_PHONEMES_DICT[token]])
        return embedding(phoneme_idx).detach()
    else:
        print("WARN: NOT A SPECIAL TOKEN")
        return get_phoneme_embeddings(token)


def g2p_tokenwise(sentence):
    phonemes = []
    for word in sentence.split(" "):
        phonemes.append(g2p_text(word))
    return phonemes
