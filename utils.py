import os
import torch
from tqdm import tqdm
from torch.nn import functional as F

def load_embeddings(path):
    """
    Load the embeddings from the given path
    :param path: Path to the embeddings
    :return: Embeddings and labels
    """
    embeddings_list = []
    labels_list = []
    for file in tqdm(os.listdir(path)):
        embedding_with_label = torch.load(os.path.join(path, file))
        embedding = embedding_with_label[:-1]
        label = embedding_with_label[-1]
        embeddings_list.append(embedding)
        labels_list.append(label)
    return torch.stack(embeddings_list), torch.stack(labels_list)


def load_waveforms(path, max_seq_length):
    """
    Load the waveforms from the given path
    :param path: Path to the waveforms
    :param max_seq_length: Max sequence length
    :return: Waveforms and labels
    """
    waveforms_list = []
    labels_list = []
    for file in tqdm(os.listdir(path)):
        waveform_with_label = torch.load(os.path.join(path, file))
        waveform = waveform_with_label[:-1]

        # Pad the waveforms to the max sequence length
        if len(waveform) < max_seq_length:
            pad_length = max_seq_length - len(waveform)
            waveform = F.pad(waveform, (0, pad_length))

        label = waveform_with_label[-1]
        waveforms_list.append(waveform)
        labels_list.append(label)

    return torch.stack(waveforms_list), torch.stack(labels_list)

def train_classifier(classifier, criterion, optimizer, num_epochs, train_embeddings, train_labels, valid_embeddings, valid_labels, device):
    """
    Train the classifier
    :param classifier: Classifier
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param num_epochs: Number of epochs
    :param train_embeddings: Train embeddings
    :param train_labels: Train labels
    :param valid_embeddings: Validation embeddings
    :param valid_labels: Validation labels
    :param device: Device
    :return: Train losses, validation losses and validation accuracies
    """
    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(num_epochs):
        classifier.train()
        inputs, targets = train_embeddings.to(device), train_labels.to(device, dtype=torch.long)
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_losses.append(train_loss)

        classifier.eval()
        with torch.no_grad():
            inputs, targets = valid_embeddings.to(device), valid_labels.to(device, dtype=torch.long)
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            valid_loss = loss.item()
            valid_losses.append(valid_loss)
            accuracy = torch.mean((torch.argmax(outputs, dim=1) == targets).float())
            valid_accs.append(accuracy.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs} Training Loss: {train_losses[-1]} Validation Loss: {valid_losses[-1]} Validation Accuracy: {valid_accs[-1] * 100:.2f}%')

    return train_losses, valid_losses, valid_accs


def test_classifier(classifier, test_embeddings, test_labels, device):
    """
    Test the classifier
    :param classifier: Classifier
    :param test_embeddings: Test embeddings
    :param test_labels: Test labels
    :param device: Device
    :return: Test accuracy
    """
    classifier.eval()
    with torch.no_grad():
        inputs, targets = test_embeddings.to(device), test_labels.to(device, dtype=torch.long)
        outputs = classifier(inputs)
        test_acc = torch.mean((torch.argmax(outputs, dim=1) == targets).float()).item()
        print(f'Test accuracy: {test_acc * 100:.2f}%')

    return test_acc


def train_transformer(model, optimizer, num_epochs, train_loader, val_loader, device):
    """
    Train the transformer
    :param model: Transformer
    :param optimizer: Optimizer
    :param num_epochs: Number of epochs
    :param train_loader: Train loader
    :param val_loader: Validation loader
    :param device: Device
    :return: Train losses, validation losses and validation accuracies
    """
    train_losses = []
    valid_losses = []
    valid_accs = []
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
            
        train_losses.append(running_loss / len(train_loader))
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
                
        valid_losses.append(val_loss / len(val_loader))
        val_accuracy = correct_predictions / len(val_loader.dataset)
        valid_accs.append(val_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs} Training Loss: {train_losses[-1]} Validation Loss: {valid_losses[-1]} Validation Accuracy: {valid_accs[-1] * 100:.2f}%')

    return train_losses, valid_losses, valid_accs


def test_transformer(model, test_loader, device):
    """
    Test the transformer
    :param model: Transformer
    :param test_loader: Test loader
    :param device: Device
    :return: Test accuracy
    """
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for inputs, mask, labels in test_loader:

            outputs = model(inputs, attention_mask=mask, labels=labels)
            predicted = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            
    test_acc = correct_predictions / len(test_loader.dataset)
    print(f'Test accuracy: {test_acc * 100:.2f}%')
    return test_acc