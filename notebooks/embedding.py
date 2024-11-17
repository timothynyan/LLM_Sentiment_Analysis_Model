import torch
vocab_size = 50257
output_dim = 256
max_length = 52

def get_embedding_layer(vocab_size, output_dim, dataloader):
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)

    embeddings = []
    for batch in dataloader:
        input_ids, labels = batch
        token_embeddings = token_embedding_layer(input_ids)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))
        embeddings.append(token_embeddings + pos_embeddings)

    return embeddings