import importlib
import tiktoken
from torch.utils.data import Dataset, DataLoader


#Local Imports
from dataloader import load_text_files
from data import Data

FILE_DIR = "../database"


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
