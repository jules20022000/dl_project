from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import soundfile as sf
import io
import speech_recognition as sr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Allow all origins for the sake of simplicity; you should restrict this in a production environment

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to a specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount the "static" directory as a root directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and model and load the weights
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=25).to(device)
model.load_state_dict(torch.load("./results/bert_large.pt"), strict=False)

id_to_prompt = {0: 'Emotional pain', 1: 'Hair falling out', 2: 'Heart hurts', 3: 'Infected wound', 4: 'Foot ache', 5: 'Shoulder pain', 6: 'Injury from sports', 7: 'Skin issue', 8: 'Stomach ache', 9: 'Knee pain', 10: 'Joint pain', 11: 'Hard to breath', 12: 'Head ache', 13: 'Body feels weak', 14: 'Feeling dizzy', 15: 'Back pain', 16: 'Open wound', 17: 'Internal pain', 18: 'Blurry vision', 19: 'Acne', 20: 'Muscle pain', 21: 'Neck pain', 22: 'Cough', 23: 'Ear ache', 24: 'Feeling cold'}


@app.get("/")
# Serve the index.html file
async def main():
    return FileResponse("templates/index.html")

# Your existing route handlers
@app.post("/record/start")
async def start_recording():
    # Placeholder for starting the recording process
    # You can implement your recording logic here
    return {"message": "Recording started"}

@app.post("/record/stop")
async def stop_recording(file: UploadFile = File(...)):
    # Save the audio data to a .wav file
    audio_bytes = await file.read()
    with io.BytesIO(audio_bytes) as buf:
        data, rate = sf.read(buf, dtype='int16')
        sf.write("temp.wav", data, rate)
    return {"message": "Recording stopped and saved as recorded_audio.wav"}

@app.get("/transcribe")
async def transcribe_recorded_audio():
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
            phrase = recognizer.recognize_google(audio_data)
            print(phrase)
            inputs = tokenizer(phrase, return_tensors="pt")
            logits = model(inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)).logits
            preds = torch.argmax(logits, dim=1)
            return {"transcription": phrase, "prediction": id_to_prompt[preds.item()]}
    except sr.UnknownValueError:
        return {"message": "Speech Recognition could not understand audio"}
    except sr.RequestError as e:
        return {"message": f"Could not request results from Google Speech Recognition service; {e}"}

