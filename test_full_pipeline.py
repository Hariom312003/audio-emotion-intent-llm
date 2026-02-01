from inference.asr import transcribe_audio
from inference.emotion import detect_emotion
from inference.intent import detect_intent

audio_path = "data/raw_audio/sample_fixed.wav" # put a wav file here

print("🎙 Transcribing audio...")
text = transcribe_audio(audio_path)
print("📝 Transcript:", text)

print("\n😄 Emotion detection...")
emotion = detect_emotion(text)
print("Emotion:", emotion)

print("\n🎯 Intent detection...")
intent = detect_intent(text)
print("Intent:", intent)
