from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
dataset = load_dataset(
    "csv",
    data_files="data/intent_dataset/intent_large.csv"
)

# --------------------------------------------------
# 2. Encode labels (string -> int)
# --------------------------------------------------
labels = sorted(set(dataset["train"]["label"]))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["labels"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)

# --------------------------------------------------
# 3. Tokenizer
# --------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=True
    )

dataset = dataset.map(tokenize, batched=True)

# --------------------------------------------------
# 4. Remove unused columns (CRITICAL)
# --------------------------------------------------
dataset = dataset.remove_columns(["text", "label"])

# --------------------------------------------------
# 5. Model
# --------------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# --------------------------------------------------
# 6. Data collator (handles padding)
# --------------------------------------------------
data_collator = DataCollatorWithPadding(tokenizer)

# --------------------------------------------------
# 7. Training configuration
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="models/intent",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none"
)

# --------------------------------------------------
# 8. Trainer (NO tokenizer argument!)
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

# --------------------------------------------------
# 9. Train
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# 10. Save model & tokenizer
# --------------------------------------------------
model.save_pretrained("models/intent")
tokenizer.save_pretrained("models/intent")

