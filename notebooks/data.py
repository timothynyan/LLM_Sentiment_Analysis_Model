import pandas as pd
import torch
from torch.utils.data import Dataset
from attention_mechanism import MultiHeadAttention


class Data(Dataset):
    def __init__(self, train, tokenizer, max_length=52, d_model=256, num_heads=8, dropout=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Create DataFrame
        self.train = self.create_dataframe(train["pos"], train["neg"])
        
        # Tokenize and pad the training data
        self.train["input_ids"] = self.train["text"].apply(self.tokenize_and_pad)
        
        # Convert input_ids and sentiment to tensors
        self.input_ids = torch.stack(self.train["input_ids"].tolist())
        self.output = torch.tensor(self.train["sentiment"].apply(lambda x: 1 if x == "pos" else 0).tolist())
        
        # Placeholder for embeddings and attention outputs
        self.embeddings = None
        self.attention_outputs = None

        # Initialize the multi-head attention mechanism
        self.multihead_attention = MultiHeadAttention(d_model, d_model, max_length, dropout, num_heads)

    def create_dataframe(self, pos_data, neg_data):
        pos_df = pd.DataFrame({'text': pos_data, 'sentiment': 'pos'})
        neg_df = pd.DataFrame({'text': neg_data, 'sentiment': 'neg'})
        return pd.concat([pos_df, neg_df], ignore_index=True)
    
    def tokenize_and_pad(self, text):
        # Tokenize the text
        input_ids = self.tokenizer.encode(text)
        
        # Pad the input_ids
        padded_input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        
        # Truncate if necessary
        padded_input_ids = padded_input_ids[:self.max_length]
        
        # Convert to tensor
        return torch.tensor(padded_input_ids)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output[idx]
    
    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
        self.attention_outputs = [self.multihead_attention(embed) for embed in embeddings]
