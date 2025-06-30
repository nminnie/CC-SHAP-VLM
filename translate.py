#!/usr/bin/env python3

import time
import sys
import json
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
    text_lower = text.strip().lower()
    conditions = [
        text_lower == "হ্যাঁ" and src_lang == "bn",
        text_lower == "ya" and src_lang == "id",
        text_lower == "네" and src_lang == "ko",
        text_lower == "да" and src_lang == "ru"
    ]
    if any(conditions):
        return "Yes"

    tokenizer.src_lang = lang_codes[src_lang]
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_codes[tgt_lang]])
    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    return translation


def evaluate_translated_prediction(translated_prediction, correct_answer):
    translated_prediction = translated_prediction.lower()
    correct_answer = correct_answer.lower()

    return 1 if correct_answer in translated_prediction else 0


def translate_model_preds(model, tokenizer, results_file, src_lang, tgt_lang="en"):
    accuracy = 0
    with open(results_file) as f:
        results = json.load(f)

    for k, result in results.items():
        if result["accuracy"] == 1:
            accuracy += result["accuracy"]
            continue

        question = result["question"]
        prediction = result["prediction"]
        correct_answer = result["correct_answer"]

        if src_lang is None:
            src_lang = result["prediction_lang"]

        if src_lang == "" or src_lang == tgt_lang:
            accuracy += result["accuracy"]
            continue

        answer_marker = "Answer: "
        context = f"Question: {question} | Answer: {prediction}"
        translation = translate_text(model, tokenizer, context, src_lang, tgt_lang)
        if answer_marker in translation:
            translation = translation.split(answer_marker)[-1]
        else:
            translation = translate_text(model, tokenizer, prediction, src_lang, tgt_lang)
        result["translated_prediction"] = translation

        accuracy_sample = evaluate_translated_prediction(translation, correct_answer)
        result["accuracy"] = accuracy_sample
        accuracy += accuracy_sample

        results[k] = result

    print(f"Accuracy: {accuracy*100/len(results):.2f}")
    return results


if __name__ == "__main__":
    model_path = "/work/tc067/tc067/s2737499/.cache/huggingface/hub/models--facebook--nllb-200-3.3B/snapshots/1a07f7d195896b2114afcb79b7b57ab512e7b43e"
    model, tokenizer = load_translation_model(model_path)

    results_file = sys.argv[1]
    src_lang = sys.argv[2] if len(sys.argv) > 2 else None
    tgt_lang = "en"

    print(f"Translating predictions from {src_lang} to {tgt_lang}: {results_file}")
    start_time = time.time()
    translated_results = translate_model_preds(model, tokenizer, results_file, src_lang, tgt_lang)
    run_time = time.time() - start_time
    print(f"\nTranslation ran for {run_time // 3600 % 24:.2f} hours, {run_time // 60 % 60:.2f} minutes, {run_time % 60:.2f} seconds.")

    filename = results_file.replace(".json", "_translated.json")
    print(f"Saving translated predictions to: {filename}")
    with open(filename, 'w') as file:
        json.dump(translated_results, file)
