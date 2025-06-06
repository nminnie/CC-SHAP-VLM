import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

lang_codes = {
    "bn": "ben_Beng",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "id": "ind_Latn",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans"  # Simplified Chinese
}

model_path = "/work/tc067/tc067/s2737499/.cache/huggingface/hub/models--facebook--nllb-200-1.3B/snapshots/b0de46b488af0cf31749cd8da5ed3171e11b2309"
def load_translation_model(model_name=model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    return model, tokenizer


def translate_text(model, tokenizer, text, src_lang, tgt_lang="en"):
    tokenizer.src_lang = lang_codes[src_lang]
    inputs = tokenizer(text, return_tensors="pt")
    output = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_codes[tgt_lang]])
    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    return translation
