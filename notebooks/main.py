import importlib
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch


#Local Imports
from dataloader import load_text_files
from data import Data
from embedding import get_embedding_layer

FILE_DIR = "../database"

#Constants
MAX_LENGTH = 52
CONTEXT_LENGTH = MAX_LENGTH
VOCAB_SIZE = 50257
OUTPUT_DIM = 256



train = load_text_files(f"{FILE_DIR}/train")

tokenizer = tiktoken.get_encoding("gpt2")
database = Data(train, tokenizer)

dataloader = DataLoader(database, batch_size=4, shuffle=True)
#%%
#Checking the batches
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
#%%
# Embedding the data into a 256 dimension space
embeddings = get_embedding_layer(VOCAB_SIZE, OUTPUT_DIM, dataloader)
database.set_embedding(embeddings)