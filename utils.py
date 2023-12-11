import os
import wave
import torch
import pyttsx3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import speech_recognition as sr
from transformers import pipeline
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix


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

    # Move the tensor to the CPU before using it with confusion_matrix
    outputs_cpu = torch.argmax(outputs.cpu(), dim=1)
    
    cm = confusion_matrix(targets.cpu(), outputs_cpu)

    return test_acc, cm


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

    all_labels = torch.tensor([], dtype=torch.long, device=device)
    all_predicted = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        for inputs, mask, labels in test_loader:

            outputs = model(inputs, attention_mask=mask, labels=labels)
            predicted = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_predicted = torch.cat((all_predicted, predicted), dim=0)
            
    test_acc = correct_predictions / len(test_loader.dataset)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

    # Move the tensor to the CPU before using it with confusion_matrix
    cm = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    
    return test_acc, cm

def test_conversion(phrases, predicted):
    """
    Calculate overall accuracy of the predicted phrases.
    :param phrases: List of ground truth phrases
    :param predicted: List of predicted phrases
    :return: Overall accuracy as a percentage
    """
    total_phrases = len(phrases)
    
    # Ensure both lists have the same length
    if total_phrases != len(predicted):
        raise ValueError("Number of predicted phrases must match the number of ground truth phrases.")
    
    correct_phrases = sum(1 for phrase, pred in zip(phrases, predicted) if phrase.lower() == pred.lower())
    
    overall_accuracy = correct_phrases / total_phrases

    return overall_accuracy


def plot_model_performance(path, title):
    """
    Plot both loss and accuracy in two subplots side by side.
    :param path: Path to the data files
    """
    # Load data
    try:
        train_losses = np.load(os.path.join(path, 'tr_losses.npy'))
        valid_losses = np.load(os.path.join(path, 'val_losses.npy'))
        valid_accs = np.load(os.path.join(path, 'val_accs.npy'))
    except:
        train_losses = None
        valid_losses = None
        valid_accs = None
    test_acc = np.load(os.path.join(path, 'test_acc.npy'))
    cm = np.load(os.path.join(path, 'cm.npy'))

    # Create subplots with a specified figsize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust the figsize as needed

    # Plot losses with a wider subplot
    if train_losses is not None and valid_losses is not None:
        axes[0].plot(train_losses, label='Training loss')
        axes[0].plot(valid_losses, label='Validation loss')
        axes[0].set_title('Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(frameon=False)
    else:
        axes[0].axis('off')

    # Plot accuracies
    if valid_accs is not None:
        axes[1].plot(valid_accs, label='Validation accuracy')
    axes[1].axhline(y=test_acc, color='g', linestyle='-', label='Test accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(frameon=False)

    # Plot confusion matrix
    im = axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[2].set_title('Confusion matrix')
    axes[2].set_ylabel('True label')
    axes[2].set_xlabel('Predicted label')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Count', rotation=270, labelpad=15)

    # Adjust layout and show the plot
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def speech_to_wav(file_path='sample.wav', sample_width=2, sample_rate=44100):
    """
    Record audio using the microphone and save it to a .wav file.
    :param file_path: Path to save the .wav file
    :param sample_width: Number of bytes per sample (1 for 8-bit, 2 for 16-bit, etc.)
    :param sample_rate: Number of samples per second (e.g., 44100 Hz)
    """
    r = sr.Recognizer()

    print("Recording...")

    # Exception handling to handle
    # exceptions at the runtime
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            # listens for the user's input 
            audio2 = r.listen(source2)

            # save the audio data to a .wav file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio2.frame_data)

            print(f"Recording saved as {file_path}")

    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    except sr.UnknownValueError:
        print("Unknown error occurred")


def speech_to_text():
    """
    Convert speech to text using the microphone.
    :return: Text
    """
    r = sr.Recognizer()

    print("Recording...")

    # Exception handling to handle
    # exceptions at the runtime
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            # listens for the user's input 
            audio2 = r.listen(source2)

            # Using google speech recognition
            text = r.recognize_google(audio2)
            return text

    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    except sr.UnknownValueError:
        print("Unknown error occurred")



def transcribe_wav_to_text(file_path, asr_pipeline):
    """
    Transcribe the audio file to text using transformers library
    :param file_path: Path to the audio file
    :return: Text
    """
    transcriptions = asr_pipeline(file_path)

    if transcriptions:
        return transcriptions['text'].lower()
    else:
        return None