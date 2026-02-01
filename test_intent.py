from inference.intent import detect_intent

tests = [
    "I want to cancel my order",
    "Where is my delivery",
    "Please help me with payment",
    "I am very happy with this service"
]

for t in tests:
    print(f"{t} -> {detect_intent(t)}")
