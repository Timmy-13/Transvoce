import torch

class TransvoceModel(torch.nn.Module):
    def __init__(self, conformer, attention, decoder, enc_att_proj):
        super().__init__()
        self.encoder = conformer
        self.attention = attention
        self.decoder = decoder
        self.enc_att_proj = enc_att_proj

    def forward(self, spec, gtp_embeddings):
        model_output = self.encoder(spec)
        expanded_output = self.enc_att_proj(model_output)

        total_phoneme_loss = 0
        for gtp in gtp_embeddings:
            attended = self.attention(gtp, context=expanded_output)
            pred = self.decoder(gtp, attended)
            loss = torch.nn.MSELoss()(pred.squeeze(0), gtp.squeeze(0))
            total_phoneme_loss += loss

        avg_loss = total_phoneme_loss / len(gtp_embeddings)
        return model_output, avg_loss
