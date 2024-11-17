import pandas as pd

class Data:
    def __init__(self, test, train, tokenizer):
        self.tokenizer = tokenizer
        
        self.test = self.create_dataframe(test["pos"], test["neg"])
        self.train = self.create_dataframe(train["pos"], train["neg"])
        
        # Tokenize the training data
        self.train["input_ids"] = self.train["text"].apply(self.tokenizer.encode)
        
        # Tokenize the test data
        self.test["input_ids"] = self.test["text"].apply(self.tokenizer.encode)
        
        # Pad the sequences
        self.train["input_ids"] = self.pad_input_ids(self.train["input_ids"])
        self.test["input_ids"] = self.pad_input_ids(self.test["input_ids"])

    def create_dataframe(self, pos_data, neg_data):
        pos_df = pd.DataFrame({'text': pos_data, 'sentiment': 'pos'})
        neg_df = pd.DataFrame({'text': neg_data, 'sentiment': 'neg'})
        df = pd.concat([pos_df, neg_df], ignore_index=True)
        return df
    
    def pad_input_ids(self, input_ids):
        # Determine the maximum length of the input_ids
        max_length = max([len(ids) for ids in input_ids])
        # Pad the input_ids
        padded_input_ids = [ids + [0]*(max_length - len(ids)) for ids in input_ids]
        return padded_input_ids