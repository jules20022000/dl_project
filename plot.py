import matplotlib.pyplot as plt
import numpy as np 
import os

bert_path = "./results/base_bert"
large_bert_path = "./results/large_bert"
hubert_path = "./results/hubert"

def plot_results(path):
    test_acc = np.load(os.path.join(path, "test_acc.npy"))
    tr_losses = np.load(os.path.join(path, "tr_losses.npy"))
    val_accs = np.load(os.path.join(path, "val_accs.npy"))
    val_losses = np.load(os.path.join(path, "val_losses.npy"))
    epochs = np.arange(1, len(val_losses) + 1)
    
    # create and save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, tr_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))

    # create and save accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accs, label="Validation accuracy")
    # horizontal line for test accuracy
    plt.hlines(test_acc, 0, len(epochs), label="Test accuracy", colors="r")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(path, "accuracy.png"))
    
plot_results(bert_path)
plot_results(large_bert_path)
plot_results(hubert_path)