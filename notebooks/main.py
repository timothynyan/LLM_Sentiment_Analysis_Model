import importlib
import tiktoken

#Local Imports
from dataloader import load_text_files
from data import Data

FILE_DIR = "../database"


train = load_text_files(f"{FILE_DIR}/train")
test = load_text_files(f"{FILE_DIR}/test")

tokenizer = tiktoken.get_encoding("gpt2")
database = Data(test, train, tokenizer, 512)
