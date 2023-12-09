import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

record_df = pd.read_csv("./data/overview-of-recordings.csv")


# Filter data for each split
train_df = record_df[record_df['split'] == 'train']
valid_df = record_df[record_df['split'] == 'validate']
test_df = record_df[record_df['split'] == 'test']

print("Nb of tr samples :", len(train_df))
print("Nb of val samples :", len(valid_df))
print("Nb of test samples :", len(test_df))


# create dictionnary to map each prompt to a number
prompt_to_id = {prompt: i for i, prompt in enumerate(record_df.prompt.unique())}

# create a new column label from the prompt column
record_df["label"] = record_df["prompt"].apply(lambda x: prompt_to_id[x])


### Classification using directly the transformer and not the embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used : ", device)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(prompt_to_id)).to(device)

train_df = record_df[record_df['split'] == 'train']
valid_df = record_df[record_df['split'] == 'validate']
test_df = record_df[record_df['split'] == 'test']

train_encodings = tokenizer(list(train_df['phrase']), truncation=True, padding=True)
val_encodings = tokenizer(list(valid_df['phrase']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_df['phrase']), truncation=True, padding=True)

train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(train_df['label'].values))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']), torch.tensor(val_encodings['attention_mask']), torch.tensor(valid_df['label'].values))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), torch.tensor(test_df['label'].values))

batch_size = 500

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Set up training parameters
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
epochs = 15

# Training loop
tr_losses = []
val_losses = []
val_accs = []
for epoch in range(epochs):
    model.train()
    tr_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        training_loss = outputs.loss
        training_loss.backward()
        optimizer.step()
        
        tr_loss += training_loss.item()
        
        print(f'Batch: {i + 1}/{len(train_loader)}, Tr. Loss:  {training_loss.item()}', end='\r')
    
    tr_losses.append(tr_loss / len(train_loader))  
        
    # Evaluation on the validation set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            validation_loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            
            val_loss += validation_loss.item()
            
    val_losses.append(val_loss / len(val_loader))    
    accuracy = correct / total
    val_accs.append(accuracy)
    print(f'Epoch {epoch + 1}/{epochs}, Tr. Loss:  {tr_losses[-1]}, Val. Loss:  {val_losses[-1]}, Val. Accuracy: {val_accs[-1] * 100:.2f}%')


# save the model 
save_path = "./bert.pt"
torch.save(model.state_dict(), save_path)

# save the losses and accs 
np.save('./results/base_bert/tr_losses.npy', np.array(tr_losses))
np.save('./results/base_bert/val_losses.npy', np.array(val_losses))
np.save('./results/base_bert/val_accs.npy', np.array(val_accs))

# TEST the model 
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct_test += (predictions == labels).sum().item()
        total_test += len(labels)

accuracy_test = correct_test / total_test
print(f'Test Accuracy: {accuracy_test}')
np.save('./results/base_bert/test_acc.npy', np.array(accuracy_test))


