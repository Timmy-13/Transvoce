import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Define Decoder-Only Transformer
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt, memory):
        tgt_emb = self.embedding(tgt)
        memory_emb = self.embedding(memory)  # Ensure memory is embedded

        # Apply positional encoding
        tgt_emb = self.positional_encoding(tgt_emb)
        memory_emb = self.positional_encoding(memory_emb)

        output = self.transformer_decoder(tgt_emb, memory_emb)
        return self.fc_out(output)

# Vocabulary setup (Fixed duplicate "Saim" and ensured all words exist)
vocab = {"<sos>": 0, "<eos>": 1, "What": 2, "is": 3, "my": 4, "name": 5, "Saim": 6,
         "Who": 7, "are": 8, "Hamna": 9, "and": 10, "Tahirah": 11, "Fatimah": 12,
         "Good": 13, "frens": 14, "who": 15, "will": 16, "offer": 17, "coffee": 18, "the":19, "drink":20}
vocab_size = len(vocab)
reverse_vocab = {v: k for k, v in vocab.items()}

def encode(sentence):
    return torch.tensor([vocab[word] for word in sentence.split()], dtype=torch.long)

def decode(tokens):
    return " ".join([reverse_vocab[token.item()] for token in tokens])

# Training Data (Ensured all words are in vocab)
training_data = [
    ("What is my name", "<sos> Saim <eos>"),
    ("What is the drink", "<sos> Good frens who will offer Saim coffee <eos>"),
    ("What is my drink", "<sos> coffee <eos>")
]

# Model setup (Increased model size)
d_model = 128  # Increased for better capacity
nhead = 4  # Adjusted for new d_model
num_layers = 2
model = DecoderOnlyTransformer(vocab_size, d_model, nhead, num_layers)

# Optimizer (Lowered learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(1000):
    total_loss = 0
    for input_sentence, output_sentence in training_data:
        optimizer.zero_grad()
        input_tensor = encode(input_sentence).unsqueeze(1)  # (seq_len, batch)
        target_tensor = encode(output_sentence).unsqueeze(1)
        output = model(target_tensor[:-1], input_tensor)  # Keep <sos> in target
        loss = criterion(output.view(-1, vocab_size), target_tensor[1:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(training_data):.4f}")

# Inference
def infer(prompt):
    model.eval()
    with torch.no_grad():
        input_tensor = encode(prompt).unsqueeze(1)
        tgt = torch.tensor([[vocab["<sos>"]]], dtype=torch.long)  # (1, 1)
        
        outputs = []
        for _ in range(10):  # Limit output length
            output = model(tgt, input_tensor).argmax(dim=-1)[-1, 0]  # Last token
            outputs.append(output.item())
            if output.item() == vocab["<eos>"]:
                break
            tgt = torch.cat([tgt, output.view(1, 1)], dim=0)
        
        return decode(torch.tensor(outputs))

print("Response:", infer("What is my name"))
print("Response:", infer("What are the drink"))
print("Response:", infer("What is my drink"))
