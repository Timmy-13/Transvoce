from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.embedding_in = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.embedding_out = nn.Embedding(output_dim, hidden_dim, padding_idx=0)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        emb_src = self.embedding_in(src)
        emb_tgt = self.embedding_out(tgt)
        _, (hidden, cell) = self.encoder(emb_src)
        out, _ = self.decoder(emb_tgt, (hidden, cell))
        return self.fc(out)