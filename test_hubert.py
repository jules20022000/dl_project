import torch
import librosa
import torch
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from transformers import logging as hf_logging
import pandas as pd

hf_logging.set_verbosity_error()

record_df = pd.read_csv("./data/overview-of-recordings.csv")

prompt_to_id = {prompt: i for i, prompt in enumerate(record_df.prompt.unique())}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_length = 295730


# Set the path to the saved model
model_path = "./hubert.pt"

# Load pre-trained Hubert model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks", num_labels=len(prompt_to_id), ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(model_path), strict=False)
model.to(device)
model.eval()

# Function to predict the label for a given audio file
def predict_label(audio_file_path):
    # Load and process the audio file
    audio_input, sr = librosa.load(audio_file_path, sr=16000)
    audio_features = feature_extractor(audio_input, sampling_rate=sr, padding=True, return_tensors="pt", max_length=max_seq_length, truncation=True)
    
    input_values = audio_features["input_values"].squeeze()
    attention_mask = audio_features["attention_mask"].squeeze()

    # Pad features to the maximum sequence length
    pad_size = max_seq_length - input_values.size(0)
    input_values = torch.nn.functional.pad(input_values, (0, pad_size))
    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size))

    # Make the prediction
    with torch.no_grad():
        input_values = input_values.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        output = model(input_values, attention_mask=attention_mask)

    predicted_label = torch.argmax(output.logits, dim=1).item()
    return predicted_label

# Replace 'path_to_test_file.wav' with the path to your test WAV file
test_file_path = 'test2.wav'
predicted_label = predict_label(test_file_path)

# Convert the predicted label back to the original prompt
id_to_prompt = {i: prompt for prompt, i in prompt_to_id.items()}
predicted_prompt = id_to_prompt[predicted_label]

print(f"The model predicts the prompt for the given audio file as: \n {predicted_prompt}")
