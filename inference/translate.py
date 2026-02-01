import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text

    tokenizer.src_lang = SRC_LANG

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

