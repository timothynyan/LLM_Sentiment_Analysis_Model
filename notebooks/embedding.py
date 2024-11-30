import torch

def get_embedding_layer(vocab_size, output_dim, dataloader, max_length, device='cpu'):
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim).to(device)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim).to(device)

    # Precompute positional embeddings
    pos_embeddings = pos_embedding_layer(torch.arange(max_length, device=device)).unsqueeze(0)

    for batch in dataloader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        token_embeddings = token_embedding_layer(input_ids)
        # Use precomputed positional embeddings
        yield token_embeddings + pos_embeddings[:, :input_ids.size(1), :]
