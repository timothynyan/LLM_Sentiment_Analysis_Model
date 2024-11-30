import torch
from torch import nn
from transformer import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], 2)  # Output dimension is 2 for binary classification

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # Shape: [batch_size, seq_len, emb_dim]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # Shape: [seq_len, emb_dim]
        pos_embeds = pos_embeds.unsqueeze(0).expand(batch_size, seq_len, -1)  # Broadcast to [batch_size, seq_len, emb_dim]
        x = tok_embeds + pos_embeds  # Shape: [batch_size, seq_len, emb_dim]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  # Shape: [batch_size, seq_len, emb_dim]
        x = self.final_norm(x[:, -1, :])  # Use only the last token representation (Shape: [batch_size, emb_dim])
        logits = self.out_head(x)  # Shape: [batch_size, 2]
        return logits
