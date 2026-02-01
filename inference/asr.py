import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model + processor (NO pipeline)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.eval()

def transcribe_audio(audio_path: str) -> str:
    """
    Robust Whisper transcription without HF pipeline bugs
    """
    # Load audio
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Whisper expects 16kHz
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Prepare input
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)

    # Decode text
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription

