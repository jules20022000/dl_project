# Voice-Driven Disease Classification: A Deep Learning Approach

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import os
from tqdm import tqdm

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


record_df = pd.read_csv("./data/overview-of-recordings2.csv")


# Filter data for each split
train_df = record_df[record_df['split'] == 'train']
valid_df = record_df[record_df['split'] == 'validate']
test_df = record_df[record_df['split'] == 'test']


# create dictionnary to map each prompt to a number
prompt_to_id = {prompt: i for i, prompt in enumerate(record_df.prompt.unique())}

# create a new column label from the prompt column
record_df["label"] = record_df["prompt"].apply(lambda x: prompt_to_id[x])


### Classification using directly the transformer and not the embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used : ", device)

tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-70b", token="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")
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-70b", token="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")


train_df = record_df[record_df['split'] == 'train']
valid_df = record_df[record_df['split'] == 'validate']
test_df = record_df[record_df['split'] == 'test']

train_encodings = tokenizer(list(train_df['phrase']), truncation=True, padding=True)
val_encodings = tokenizer(list(valid_df['phrase']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_df['phrase']), truncation=True, padding=True)

train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(train_df['label'].values))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']), torch.tensor(val_encodings['attention_mask']), torch.tensor(valid_df['label'].values))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), torch.tensor(test_df['label'].values))

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up training parameters
optimizer = optim.AdamW(model.parameters(), lr=1e-5)


print("OK ! ")