import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import Data
from dataloader import load_text_files

FILE_DIR = "../database"
train = load_text_files(f"{FILE_DIR}/train")

pos_count = len(train["pos"])
neg_count = len(train["neg"])

pos_word_count = [len(review.split()) for review in train["pos"]]
neg_word_count = [len(review.split()) for review in train["neg"]]

mean_pos_word_count = np.mean(pos_word_count)
mean_neg_word_count = np.mean(neg_word_count)

word_count_df = pd.DataFrame(
    {
        "sentiment": ["positive", "negative"],
        "count": [pos_count, neg_count],
        "avg_word_count": [mean_pos_word_count, mean_neg_word_count],
    }
)
bin_edges = np.arange(0, 1401, 20)

plt.hist(pos_word_count, bins=bin_edges, alpha=0.5, label="positive")
plt.hist(neg_word_count, bins=bin_edges, alpha=0.5, label="negative")
plt.legend(loc="upper right")
plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.xlim(0, 1400)
plt.show()
