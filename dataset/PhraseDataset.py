import torch
from torch.utils.data import Dataset

class PhraseDataset(Dataset):
    def __init__(self, phrases, labels, tokenizer, max_seq_length, device):
        self.phrases = phrases
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        label = self.labels[idx]
        
        # tokenize phrase
        encoding = self.tokenizer(phrase, return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_length)
        input_values = encoding['input_ids'].squeeze().to(self.device)
        attention_mask = encoding['attention_mask'].squeeze().to(self.device)

        # Pad features to the maximum sequence length
        pad_size = self.max_seq_length - input_values.size(0)
        input_values = torch.nn.functional.pad(input_values, (0, pad_size))
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size))

        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return input_values, attention_mask, label