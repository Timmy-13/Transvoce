import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Seq2Seq
from config import *

class UrduG2PDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=None, names=["word", "phoneme"])
        self.words = [list(w) for w in df['word']]
        self.phonemes = [list(p) for p in df['phoneme']]

        self.word_vocab = self.build_vocab(self.words)
        self.phoneme_vocab = self.build_vocab(self.phonemes)

    def build_vocab(self, sequences):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        idx = 3
        for seq in sequences:
            for char in seq:
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1
        return vocab

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = [self.word_vocab[c] for c in self.words[idx]]
        phoneme = [self.phoneme_vocab[c] for c in self.phonemes[idx]]
        word = [1] + word + [2]  # <sos>, <eos>
        phoneme = [1] + phoneme + [2]  # <sos>, <eos>
        return torch.tensor(word), torch.tensor(phoneme)

def collate_fn(batch):
    words, phonemes = zip(*batch)
    words_pad = nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=0)
    phonemes_pad = nn.utils.rnn.pad_sequence(phonemes, batch_first=True, padding_value=0)
    return words_pad, phonemes_pad


def train():
    dataset = UrduG2PDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = Seq2Seq(len(dataset.word_vocab), len(dataset.phoneme_vocab), HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    start_epoch = 0

    try:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint["model"])
        print("Loaded checkpoint, continuing training.")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save({
        "model": model.state_dict(),
        "word_vocab": dataset.word_vocab,
        "phoneme_vocab": dataset.phoneme_vocab
    }, MODEL_NAME)