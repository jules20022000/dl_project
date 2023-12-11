import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audios, labels, tokenizer, max_seq_length, device):
        self.audios = audios
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        label = self.labels[idx]

        # extract audio features
        audio_input, sr = librosa.load(audio, sr=16000)
        audio_features = self.tokenizer(audio_input, sampling_rate=sr, padding=True, return_tensors="pt", max_length=self.max_seq_length, truncation=True)
        input_values = audio_features["input_values"].squeeze().to(self.device)
        attention_mask = audio_features["attention_mask"].squeeze().to(self.device)

        # Pad features to the maximum sequence length
        pad_size = self.max_seq_length - input_values.size(0)
        input_values = torch.nn.functional.pad(input_values, (0, pad_size))
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size))

        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return input_values, attention_mask, label