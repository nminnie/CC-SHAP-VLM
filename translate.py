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

def load_translation_model(model_name="facebook/nllb-200-3.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def translate_text(model, tokenizer, text, src_lang, tgt_lang="en"):
    inputs = tokenizer(text, return_tensors="pt", src_lang=lang_codes[src_lang])
    output = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_codes[tgt_lang]])
    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    return translation
