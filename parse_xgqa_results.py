import sys
import json
from collections import defaultdict

LANG = sys.argv[1]

results_file = f"results/translated_pangea_prompt/xgqa_llava_mistral_{LANG}_200_translated.json"
if LANG == "en":
    results_file = results_file.replace("_translated", "")
with open(results_file, "r") as f:
    results = json.load(f)

lang_count = 0
accuracy = defaultdict(int)
accuracy_given_lang = defaultdict(int)
pred_lang = defaultdict(int)
for k, result in results.items():
    accuracy_sample = result["accuracy"]
    accuracy[accuracy_sample] += 1

    pred_lang_sample = result["prediction_lang"]
    if result["prediction"].strip().lower() == "no":
        pred_lang_sample = "en"
    
    pred_lang[pred_lang_sample] += 1

    if pred_lang_sample == LANG:
        lang_count += 1
        accuracy_given_lang[accuracy_sample] += 1

n_samples = 200
pred_lang = {k: (v / n_samples * 100) for k, v in pred_lang.items()}
accuracy = {k: (v / n_samples * 100) for k, v in accuracy.items()}
accuracy_given_lang = {k: (v / lang_count * 100) for k, v in accuracy_given_lang.items()}

print("Pred lang:", dict(pred_lang))
print("Accuracy:", dict(accuracy))
print(f"Accuracy for {LANG}:", dict(accuracy_given_lang))
