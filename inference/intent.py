from transformers import pipeline

intent_classifier = pipeline(
    "text-classification",
    model="models/intent"
)

def detect_intent(text: str):
    return intent_classifier(text)[0]
