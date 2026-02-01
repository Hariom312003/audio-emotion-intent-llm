# 🎙️ Multilingual Audio Emotion & Intent Detection System

An end-to-end AI system that analyzes **Hindi and English audio** to detect
**emotion and intent**, exposed through a FastAPI REST service.

---

## 🚀 Features
- Speech-to-text using Whisper
- Hindi → English translation using NLLB
- Emotion detection using transformer models
- Intent classification with a custom-trained model
- FastAPI-based REST API
- Audio normalization using FFmpeg

---

## 🧠 System Pipeline

Audio → ASR → Translation → Emotion Detection → Intent Detection → API Response

---

## 🛠 Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- FastAPI
- Whisper
- NLLB
- FFmpeg

---

## 📦 Project Structure
audio_emotion_intent_llm/
│
├── api/
│ └── app.py
├── inference/
│ ├── asr.py
│ ├── translate.py
│ ├── emotion.py
│ └── intent.py
├── models/
│ └── intent/
├── data/
│ └── raw_audio/
├── test_full_pipeline.py
├── requirements.txt
└── README.md


---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload
