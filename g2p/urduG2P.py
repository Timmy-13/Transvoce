from .resources.g2p_ur import G2p

SPECIAL_TOKENS = ["<sos>", "<pad>", "<eos>"]

g2p = G2p()

def get_phoneme_embeddings(sentence):
    phoneme_embeddings = []
    pad_embedding = g2p.get_embedding_for_special_token(SPECIAL_TOKENS[1])
    for word in sentence.split(" "):
        phonemes, embeddings = g2p.infer(word)
        phoneme_embeddings.extend(embeddings)
        phoneme_embeddings.append(pad_embedding)
    return phoneme_embeddings[:-1]

def get_special_token_embedding(token):
    if token in SPECIAL_TOKENS:
        return g2p.get_embedding_for_special_token(token)
    else:
        print("WARN: NOT A SPECIAL TOKEN")
        return get_phoneme_embeddings(token)