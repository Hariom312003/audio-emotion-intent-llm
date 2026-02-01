import random
import csv

complaint = [
    "I am not happy with the service",
    "I want a refund",
    "This product is bad",
    "I want to file a complaint",
    "Very disappointed with the service"
]

query = [
    "Where is my order",
    "What is the delivery status",
    "Can you tell me my order details",
    "When will my package arrive",
    "Track my order please"
]

request = [
    "Please help me",
    "I need assistance with payment",
    "Can you help me reset my account",
    "I need support",
    "Help me with this issue"
]

feedback = [
    "Great service",
    "Thank you very much",
    "I am happy with the service",
    "Excellent support",
    "Very satisfied"
]

rows = []

for _ in range(250):
    rows.append((random.choice(complaint), "complaint"))
    rows.append((random.choice(query), "query"))
    rows.append((random.choice(request), "request"))
    rows.append((random.choice(feedback), "feedback"))

random.shuffle(rows)

with open("data/intent_dataset/intent_large.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print("Generated", len(rows), "samples")
