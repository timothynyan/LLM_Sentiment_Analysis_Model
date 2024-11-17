import pandas as pd
import torch

class Data:
    def __init__(self, train, tokenizer, max_length=52):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.train = self.create_dataframe(train["pos"], train["neg"])
        
        # Tokenize and pad the training data
        self.train["input_ids"] = self.train["text"].apply(self.tokenize_and_pad)
        
        # Convert input_ids and sentiment to tensors
        self.input_ids = torch.stack(self.train["input_ids"].tolist())
        self.output = torch.tensor(self.train["sentiment"].apply(lambda x: 1 if x == "pos" else 0).tolist())

        self.embedding = None

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output[idx]
    

    def create_dataframe(self, pos_data, neg_data):
        pos_df = pd.DataFrame({'text': pos_data, 'sentiment': 'pos'})
        neg_df = pd.DataFrame({'text': neg_data, 'sentiment': 'neg'})
        return pd.concat([pos_df, neg_df], ignore_index=True)
    
    def tokenize_and_pad(self, text):
        input_ids = self.tokenizer.encode(text)
        
        # Pad the input_ids
        padded_input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        
        padded_input_ids = padded_input_ids[:self.max_length]
        
        return torch.tensor(padded_input_ids)
    
    def set_embedding(self, embedding):
        self.embedding = embedding
