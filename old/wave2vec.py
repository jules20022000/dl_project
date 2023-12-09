import os
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Set the path to your audio files and corresponding labels
data_dir = "./data/recordings"
train_files = set(os.listdir(os.path.join(data_dir, "train")))
valid_files = set(os.listdir(os.path.join(data_dir, "validate")))
test_files = set(os.listdir(os.path.join(data_dir, "test")))

record_df = pd.read_csv("./data/overview-of-recordings.csv")
record_df["split"] = record_df["file_name"].apply(lambda x: "train" if x in train_files else ("validate" if x in valid_files else "test"))
train_df = record_df[record_df['split'] == 'train']
valid_df = record_df[record_df['split'] == 'validate']
test_df = record_df[record_df['split'] == 'test']

# append data_dir to file names
train_files = [os.path.join(data_dir, "train", f) for f in train_df["file_name"]]
valid_files = [os.path.join(data_dir, "validate", f) for f in valid_df["file_name"]]
test_files = [os.path.join(data_dir, "test", f) for f in test_df["file_name"]]

prompt_to_id = {prompt: i for i, prompt in enumerate(record_df.prompt.unique())}

train_labels = train_df.prompt.apply(lambda x: prompt_to_id[x]).values
valid_labels = valid_df.prompt.apply(lambda x: prompt_to_id[x]).values
test_labels = test_df.prompt.apply(lambda x: prompt_to_id[x]).values

# Function to train the model
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    tr_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, mask, labels) in enumerate(train_loader):
            # Send inputs and labels to the device
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f'Batch {i+1}/{len(train_loader)} Tr. Loss: {loss.item():.4f}', end='\r')
            
        tr_losses.append(running_loss / len(train_loader))
        # Validate the model
        model.eval()
        val_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for i,  (inputs, mask, labels) in enumerate(val_loader):

                outputs = model(inputs, attention_mask=mask, labels=labels)

                loss = outputs.loss

                val_loss += loss.item()
                predicted = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                
        val_losses.append(val_loss / len(val_loader))
        val_accuracy = correct_predictions / len(val_loader.dataset)
        val_accs.append(val_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs} Training Loss: {tr_losses[-1]} Validation Loss: {val_losses[-1]} Validation Accuracy: {val_accs[-1] * 100:.2f}%')

    return tr_losses, val_losses, val_accs


# Function to test the model
def test_model(test_loader, model):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, mask, labels in test_loader:

            outputs = model(inputs, attention_mask=mask, labels=labels)

            loss = outputs.loss

            running_loss += loss.item()
            predicted = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            
    accuracy = correct_predictions / len(test_loader.dataset)
    test_loss = running_loss / len(test_loader)
    print(f'Test Loss: {test_loss} Accuracy: {accuracy * 100:.2f}%')
    return test_loss, accuracy
    
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_extractor, max_seq_length):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # extract audio features
        audio_input, sr = librosa.load(audio_path, sr=16000)
        audio_features = self.feature_extractor(audio_input, sampling_rate=sr, padding=True, return_tensors="pt", max_length=self.max_seq_length, truncation=True)
        input_values = audio_features["input_values"].squeeze().to(device)
        attention_mask = audio_features["attention_mask"].squeeze().to(device)

        # Pad features to the maximum sequence length
        pad_size = self.max_seq_length - input_values.size(0)
        input_values = torch.nn.functional.pad(input_values, (0, pad_size))
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size))

        label = torch.tensor(label, dtype=torch.long).to(device)

        return input_values, attention_mask, label

        
# Load pre-trained wave2vec model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks", num_labels=len(prompt_to_id), ignore_mismatched_sizes=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

max_seq_length = 295730

train_dataset = AudioDataset(train_files, train_labels, feature_extractor, max_seq_length)
valid_dataset = AudioDataset(valid_files, valid_labels, feature_extractor, max_seq_length)
test_dataset = AudioDataset(test_files, test_labels, feature_extractor, max_seq_length)

batch_size = 5
epochs = 15

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model
tr_losses, val_losses, val_accs = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=epochs)

# save the model 
save_path = "./wave2vec.pt"
torch.save(model.state_dict(), save_path)

# save the losses and accs 
np.save('./results/wave2vec/tr_losses.npy', np.array(tr_losses))
np.save('./results/wave2vec/val_losses.npy', np.array(val_losses))
np.save('./results/wave2vec/val_accs.npy', np.array(val_accs))

# Test the model
test_loss, test_acc = test_model(test_loader, model)
np.save('./results/wave2vec/test_loss.npy', np.array(test_loss))
np.save('./results/wave2vec/test_acc.npy', np.array(test_acc))

