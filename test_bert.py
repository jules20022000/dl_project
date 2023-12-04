import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import logging as hf_logging
import pandas as pd
from speech_to_text import speech_to_text

hf_logging.set_verbosity_error()

record_df = pd.read_csv("./data/overview-of-recordings.csv")

prompt_to_id = {prompt: i for i, prompt in enumerate(record_df.prompt.unique())}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(prompt_to_id)).to(device)

model_path = "./bert_model_large.pt"
model.load_state_dict(torch.load(model_path))
# Assuming you have a trained model stored in the variable 'model'
model.eval()

# texts = [
#     "Trying to cope with the overwhelming sorrow that engulfs me whenever thoughts of her arise.",
#     "Watching helplessly as strands of hair fall out each day, leaving me distressed and anxious.",
#     "Enduring a constant ache in my chest, as if my heart is burdened with an unexplainable sorrow.",
#     "Feeling the throbbing pain of an infected wound, with visible signs of discomfort and concern.",
#     "Struggling with each step as my foot aches persistently, making walking an arduous task.",
#     "Experiencing a sharp and persistent pain in my shoulder, limiting the movement of my arm.",
#     "Bearing the intense pain of a sports injury, each moment a reminder of the physical toll it took.",
#     "Dealing with irritated and rash-covered skin, the discomfort and itchiness becoming unbearable.",
#     "Enduring a constant stomach ache, the sensation of knots forming inside, causing great discomfort.",
#     "Coping with constant knee pain that makes bending or walking a challenging and painful endeavor.",
#     "Aching joints throughout my body, each movement accompanied by persistent and increasing pain.",
#     "Struggling to breathe with each inhale, as if the air carries a heavy burden that I must bear.",
#     "Enduring a pounding headache that refuses to subside, making focus and concentration nearly impossible.",
#     "Feeling an overall weakness and fatigue that permeates every part of my body, draining my strength.",
#     "Experiencing unexpected bouts of dizziness, making it difficult to maintain balance and focus.",
#     "Constant back pain making even the simplest movements unbearable, a persistent and nagging discomfort.",
#     "Navigating the challenges of an open wound, the pain and discomfort a constant presence.",
#     "Suffering from a deep internal pain that eludes identification, affecting both body and emotions.",
#     "Struggling with blurry vision, each moment a challenge as clarity becomes increasingly elusive.",
#     "Dealing with persistent acne that extends beyond a cosmetic concern, causing physical discomfort and affecting confidence.",
#     "Enduring sore and achy muscles, each movement a reminder of the physical strain.",
#     "Coping with the stiff and unrelenting pain in my neck, making every movement a source of discomfort.",
#     "Unable to cease the relentless coughing, each fit making it harder to breathe properly.",
#     "Enduring a throbbing earache, where every sound feels like a sharp and painful stab.",
#     "Constantly feeling cold and shivery, unable to shake off the persistent chill that envelops my body."
# ]

# NOTE : harder prompts with less info
# texts = [
#     "Trying to cope with overwhelming sorrow whenever thoughts of her arise.",
#     "Watching helplessly as strands of hair fall out each day, leaving me distressed and anxious.",
#     "Enduring a constant ache in my chest, as if my heart is burdened with an unexplainable sorrow.",
#     "Feeling the throbbing pain of an infected wound, with visible signs of discomfort and concern.",
#     "Struggling with each step as persistent aches make walking an arduous task.",
#     "Experiencing sharp and persistent discomfort, limiting the movement of my arm.",
#     "Bearing the intense pain of an injury, each moment a reminder of the physical toll it took.",
#     "Dealing with irritated and rash-covered skin, the discomfort and itchiness becoming unbearable.",
#     "Enduring a constant sensation inside, causing great discomfort.",
#     "Coping with constant discomfort that makes bending or walking a challenging endeavor.",
#     "Aching joints throughout my body, each movement accompanied by persistent and increasing discomfort.",
#     "Struggling to breathe with each inhale, as if the air carries a heavy burden that I must bear.",
#     "Enduring a pounding sensation that refuses to subside, making focus and concentration nearly impossible.",
#     "Feeling an overall weakness and fatigue that permeates every part of my body, draining my strength.",
#     "Experiencing unexpected bouts of dizziness, making it difficult to maintain balance and focus.",
#     "Constant discomfort making even the simplest movements unbearable, a persistent and nagging challenge.",
#     "Navigating the challenges of an open wound, the discomfort and pain a constant presence.",
#     "Suffering from a deep internal discomfort that eludes identification, affecting both body and emotions.",
#     "Struggling with vision challenges, each moment a challenge as clarity becomes increasingly elusive.",
#     "Dealing with persistent challenges that extend beyond cosmetic concerns, causing physical discomfort and affecting confidence.",
#     "Enduring sore and achy muscles, each movement a reminder of the physical strain.",
#     "Coping with the stiff and unrelenting discomfort, making every movement a source of challenge.",
#     "Unable to cease the relentless coughing, each fit making it harder to breathe properly.",
#     "Enduring a throbbing sensation, where every sound feels like a sharp and painful stab.",
#     "Constantly feeling cold and shivery, unable to shake off the persistent chill that envelops my body."
# ]

text1 = speech_to_text()
print("You said : {}".format(text1))
text2 = speech_to_text()
print("You said : {}".format(text2))
texts = [text1, text2]

labels = list(range(25))
acc = 0
for i, new_text in enumerate(texts):

    # Tokenize the new text
    new_text_encoding = tokenizer(new_text, truncation=True, padding=True, return_tensors='pt').to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(**new_text_encoding)

    # Get the predicted class
    predicted_class = torch.argmax(output.logits).item()

    print(f"Predicted class:  {list(prompt_to_id.keys())[predicted_class]}")
    
    if predicted_class == labels[i]:
        acc += 1
        
        
print("Accuracy : {}% ".format(acc / len(labels) * 100))