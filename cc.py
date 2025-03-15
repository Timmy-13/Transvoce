import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.3, phoneme_emb_dim=512, label_smoothing=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, phoneme_emb_dim)
        self.positional_encoding = PositionalEncoding(phoneme_emb_dim, dropout)
        
        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.label_smoothing = label_smoothing
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.embedding(tgt)  # Convert token indices to embeddings
        tgt_emb = self.positional_encoding(tgt_emb)
        
        output = self.decoder_layers(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        print(output.shape)
        logits = self.fc_out(output)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Example usage
decoder = TransformerDecoder(vocab_size=1000)  # Adjust vocab_size as needed
tgt = torch.randint(0, 1000, (2, 20))  # Example batch of target sequences
memory = torch.randn(2, 20, 512)  # Example encoder outputs
output = decoder(tgt, memory)
print(output.shape)  # Expected: (batch_size, sequence_length, vocab_size)
