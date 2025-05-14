from .config import *
from .model import Seq2Seq

class UrduG2P:
    def __init__(self):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        self.word_vocab = checkpoint["word_vocab"]
        self.phoneme_vocab = checkpoint["phoneme_vocab"]
        self.idx_to_phoneme = {v: k for k, v in self.phoneme_vocab.items()}

        self.model = Seq2Seq(len(self.word_vocab), len(self.phoneme_vocab), HIDDEN_SIZE).to(DEVICE)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()


    def infer(self, word):
        input_seq = [1] + [self.word_vocab.get(c, 0) for c in word] + [2]
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(DEVICE)

        output_seq = []
        embeddings = []

        with torch.no_grad():
            emb = self.model.embedding_in(input_tensor)
            _, (hidden, cell) = self.model.encoder(emb)
            input_token = torch.tensor([[1]]).to(DEVICE)
            output_seq = []

            for _ in range(30):
                emb_tgt = self.model.embedding_out(input_token)
                out, (hidden, cell) = self.model.decoder(emb_tgt, (hidden, cell))
                logits = self.model.fc(out[:, -1])
                pred_token = logits.argmax(dim=-1).item()
                if pred_token == 2:
                    break
                output_seq.append(self.idx_to_phoneme.get(pred_token, '?'))
                embeddings.append(emb_tgt.squeeze(0).cpu())
                input_token = torch.tensor([[pred_token]]).to(DEVICE)

        return ''.join(output_seq), embeddings

    def get_embedding_for_special_token(self, token):
        token_idx = self.phoneme_vocab[token]
        token_tensor = torch.tensor([token_idx]).to(DEVICE)
        token_embedding = self.model.embedding_out(token_tensor)
        return token_embedding.detach()