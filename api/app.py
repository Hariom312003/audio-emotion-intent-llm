from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from inference.asr import transcribe_audio
from inference.emotion import detect_emotion
from inference.intent import detect_intent
from inference.translate import translate_to_english

app = FastAPI(
    title="Audio Emotion + Intent Detection API",
    version="1.0"
)

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -------- PIPELINE --------
    # 1️⃣ ASR
    text = transcribe_audio(file_path)

    # 2️⃣ Translate Hindi -> English (safe for English too)
    translated_text = translate_to_english(text)

    # 3️⃣ Emotion + Intent
    # 🔥 Force English text for emotion model
    emotion_input = (
        translated_text
        if translated_text and translated_text.strip() != text.strip()
        else translated_text
    )

    if emotion_input:
        emotion = detect_emotion(emotion_input)
    else:
        emotion = {
            "label": "unknown",
            "score": 0.0
        }

    intent = detect_intent(translated_text)

    # 🧹 Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "transcript": text,
        "translated_text": translated_text,
        "emotion": emotion,
        "intent": intent
    }

