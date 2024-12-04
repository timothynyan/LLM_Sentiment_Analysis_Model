# %%
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from data import Data
from dataloader import load_text_files  # Import the load_text_files function
from gptmodel import GPTModel  # Import the GPTModel class
import tiktoken  # Import the tiktoken library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODEL_SAVE_PATH = "gpt_model.pth"
OPTIMIZER_SAVE_PATH = "optimizer.pth"
FILE_DIR = "../database"
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "num_heads": 4,
    "n_layers": 4,
    "d_model": 64,
    "dropout": 0.1,
    "max_length": 64,
    "qkv_bias": False,
    "emb_dim": 64,
}

# Load train data
train = load_text_files(f"{FILE_DIR}/train")

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Create the Data object
database = Data(train, tokenizer, GPT_CONFIG)

# Create the DataLoader with batch_size=1
dataloader = DataLoader(database, batch_size=1, shuffle=True)

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the model and move it to the device
model = GPTModel(GPT_CONFIG).to(device)

# Define the optimizer and loss function
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)

if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Model loaded from", MODEL_SAVE_PATH)

if os.path.exists(OPTIMIZER_SAVE_PATH):
    optimizer.load_state_dict(torch.load(OPTIMIZER_SAVE_PATH))
    print("Optimizer loaded from", OPTIMIZER_SAVE_PATH)

# Training loop
num_epochs = 10
# %%
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids)
        logits.squeeze(-1)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    torch.save(optimizer.state_dict(), OPTIMIZER_SAVE_PATH)


# %%
output_df = pd.DataFrame(
    columns=["text", "prediction", "original", "probability", "identical"]
)
model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids, labels = batch
        if labels.item() == 1:
            orginal_labels = "pos"
        else:
            orginal_labels = "neg"

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        logits = logits.squeeze(-1)

        # Convert logits to probabilities using softmax or sigmoid
        probabilities = torch.sigmoid(logits[:, 0])  # Assuming binary logits

        # Apply a threshold to classify as positive/negative
        predictions = (probabilities > 0.5).int()
        if predictions.item() == 0:
            prediction = "pos"
        else:
            prediction = "neg"
        original_text = tokenizer.decode(input_ids[0].tolist())

        # Check if the prediction is correct
        identical = prediction == orginal_labels

        new_row = pd.DataFrame(
            {
                "text": [original_text],
                "prediction": prediction,
                "original": orginal_labels,
                "probability": [probabilities.item()],
                "identical": [identical],
            }
        )

        # Concatenate the new row to the output DataFrame
        output_df = pd.concat([output_df, new_row], ignore_index=True)


# %%
# Plot the histogram of predictions and original labels to check for class imbalance and accuracy
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histogram of output_df["prediction"]
output_df["prediction"].value_counts().sort_index().plot(kind="bar", ax=axs[0])
axs[0].set_title("Prediction")
axs[0].set_xlabel("Prediction")
axs[0].set_ylabel("Count")

# Add counts as text annotations
for i, count in enumerate(output_df["prediction"].value_counts().sort_index()):
    axs[0].text(i, count, str(count), ha="center", va="bottom")

# Histogram of output_df["original_label"]
output_df["original"].value_counts().sort_index().plot(kind="bar", ax=axs[1])
axs[1].set_title("Original")
axs[1].set_xlabel("Original Label")
axs[1].set_ylabel("Count")

# Add counts as text annotations
for i, count in enumerate(output_df["original"].value_counts().sort_index()):
    axs[1].text(i, count, str(count), ha="center", va="bottom")

plt.tight_layout()
plt.show()

# %%
output_df["identical"].astype(int).hist()
plt.title("Identical Predictions")
plt.xlabel("Identical")
plt.ylabel("Count")
plt.show()


# %%
# Evaluate the model on the test data
test = load_text_files(f"{FILE_DIR}/test")
output_df_test = pd.DataFrame(
    columns=["text", "prediction", "original", "probability", "identical"]
)
database_test = Data(test, tokenizer, GPT_CONFIG)
dataloader_test = DataLoader(database_test, batch_size=1, shuffle=True)


model.eval()
with torch.no_grad():
    for batch in dataloader_test:
        input_ids, labels = batch
        if labels.item() == 1:
            orginal_labels = "pos"
        else:
            orginal_labels = "neg"

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        logits = logits.squeeze(-1)

        # Convert logits to probabilities using softmax or sigmoid
        probabilities = torch.sigmoid(logits[:, 0])  # Assuming binary logits

        # Apply a threshold to classify as positive/negative
        predictions = (probabilities > 0.5).int()
        if predictions.item() == 0:
            prediction = "pos"
        else:
            prediction = "neg"
        original_text = tokenizer.decode(input_ids[0].tolist())

        # Check if the prediction is correct
        identical = prediction == orginal_labels

        new_row = pd.DataFrame(
            {
                "text": [original_text],
                "prediction": prediction,
                "original": orginal_labels,
                "probability": [probabilities.item()],
                "identical": [identical],
            }
        )

        # Concatenate the new row to the output DataFrame
        output_df_test = pd.concat([output_df_test, new_row], ignore_index=True)


# %%
# Custom test input
while True:
    user_input = input("Enter a sentence to test (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    # Tokenize the input
    input_ids = tokenizer.encode(user_input)
    input_tensor = (
        torch.tensor(input_ids).unsqueeze(0).to(device)
    )  # Add batch dimension

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        logits = logits.squeeze(-1)

        # Convert logits to probabilities using sigmoid
        probabilities = torch.sigmoid(logits[:, 0])

        # Apply a threshold to classify as positive/negative
        prediction = "pos" if probabilities.item() > 0.5 else "neg"

        print(f"Input: {user_input}")
        print(f"Prediction: {prediction}")
        print(f"Probability: {probabilities.item():.4f}")

# %%
