from transformers import pipeline

# Emotion classification pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def detect_emotion(text: str):
    """
    Returns top emotion with score
    """
    result = emotion_classifier(text)

    # HF pipeline returns a list with one dict
    if isinstance(result, list):
        return result[0]

    # Fallback (just in case)
    return result

