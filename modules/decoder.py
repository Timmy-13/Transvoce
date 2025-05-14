from torch import nn


class PhonemeDecoder(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=4, dropout=0.3):
        super(PhonemeDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt_seq, memory, tgt_mask=None, tgt_key_padding_mask=None):
        return self.transformer_decoder(tgt_seq, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        