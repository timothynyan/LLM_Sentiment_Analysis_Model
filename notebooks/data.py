# create a class that have 2 attributes that are dictionaries of data
# and a method that returns the data in a pandas dataframe
import pandas as pd


class Data:
    def __init__(self, test, train):
        self.pos_test = test["pos"]
        self.neg_test = test["neg"]
        self.pos_train = train["pos"]
        self.neg_train = train["neg"]
        
        self.test = self.create_dataframe(self.pos_test, self.neg_test)
        self.train = self.create_dataframe(self.pos_train, self.neg_train)

    def create_dataframe(self, pos, neg):
        data = {"text": pos + neg, "label": ["pos"] * len(pos) + ["neg"] * len(neg)}
        return pd.DataFrame(data)
        
    def get_test(self):
        return self.test
    
    def get_train(self):
        return self.train
    

    
        