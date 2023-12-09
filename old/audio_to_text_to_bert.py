import os
import torch
import pandas as pd
import speech_recognition as sr
from transformers import logging as hf_logging
from transformers import BertForSequenceClassification, BertTokenizer

hf_logging.set_verbosity_error()

record_df = pd.read_csv("./data/overview-of-recordings.csv")
prompt_to_id = {prompt: i for i, prompt in enumerate(record_df.prompt.unique())}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(prompt_to_id)).to(device)

model_path = "./trained_models/bert_large.pt"
model.load_state_dict(torch.load(model_path))
# Assuming you have a trained model stored in the variable 'model'
model.eval()

record_df = record_df[(record_df['split'] == 'test') | (record_df['split'] == 'validate')]

print("Nb of test samples :", len(record_df))

# text_acc = 0
# print("Predicting labels for text only...")
# # Test the model with text only (no audio)
# for cnt, (idx, row) in enumerate(record_df.iterrows()):
#     label = row["prompt"]
#     text = row["phrase"]
#     text_encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**text_encoding)
  
#     predicted_label = torch.argmax(outputs.logits).item()	
#     if predicted_label == prompt_to_id[label]:
#         text_acc += 1
  
#     if cnt % 30 == 0:
#         print(f"{cnt}/{len(record_df)} texts were predicted", end="\r")
  
# print("Text accuracy : ", text_acc / len(record_df))
    


def transcribe_wav_to_text(wav_file_path):
    # Create a speech recognition object
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file_path) as audio_file:
        # Record the audio from the file
        audio_data = recognizer.record(audio_file)

        try:
            # Use the Google Web Speech API to recognize the audio
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")


print("Predicting labels for audio only...")
# Test the model with audio only (no text)
audio_acc = 0
audio_corrected_translated = 0
for cnt, (idx, row) in enumerate(record_df.iterrows()):
    label = row["prompt"]
    audio_file = os.path.join("./data/recordings", row["split"], row["file_name"])
    text = transcribe_wav_to_text(audio_file)
    if text is None:
        continue
    text_encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**text_encoding)
  
    audio_corrected_translated += 1
    predicted_label = torch.argmax(outputs.logits).item()	
    if predicted_label == prompt_to_id[label]:
        audio_acc += 1
  
    if cnt % 30 == 0:
        print(f"{cnt}/{len(record_df)} audios were predicted", end="\r")
        
print("Audio accuracy : ", audio_acc / audio_corrected_translated)


