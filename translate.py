import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from generation_and_prompting import evaluate_prediction


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

def load_translation_model(model_name):
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


def translate_model_preds(model, tokenizer, results_file, src_lang, tgt_lang="en"):
    with open(results_file) as f:
        results = json.load(f)

    for k, result in results.items():
        prediction = result["prediction"]
        correct_answer = result["correct_answer"]
        src_lang = result["prediction_lang"]

        if src_lang == tgt_lang:
            continue
        translation = translate_text(model, tokenizer, prediction, src_lang, tgt_lang)
        result["translated_prediction"] = translation

        accuracy = evaluate_prediction(translation, correct_answer, c_task="xgqa")
        result["accuracy"] = accuracy

        results[k] = result

    return results


if __name__ == "__main__":
    model_path = "/work/tc067/tc067/s2737499/.cache/huggingface/hub/models--facebook--nllb-200-1.3B/snapshots/b0de46b488af0cf31749cd8da5ed3171e11b2309"
    model, tokenizer = load_translation_model(model_path)

    results_file = sys.argv[1]
    tgt_lang = "en"

    translated_results = translate_model_preds(model, tokenizer, results_file, tgt_lang)

    filename = results_file.replace(".json", "_translated.json")
    with open(filename, 'w') as file:
        json.dump(translated_results, file)
