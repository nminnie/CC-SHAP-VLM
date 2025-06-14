import sys
import json
from collections import defaultdict

LANG = sys.argv[1]

results_file = f"results/translated_mm_prompt/xgqa_bakllava_{LANG}_200_translated.json"
with open(results_file, "r") as f:
    results = json.load(f)

accuracy = defaultdict(int)
pred_lang = defaultdict(int)
for k, result in results.items():
    accuracy_sample = result["accuracy"]
    accuracy[accuracy_sample] += 1

    pred_lang_sample = result["prediction_lang"]
    if result["prediction"].strip().lower() == "no":
        pred_lang_sample = "en"
    
    pred_lang[pred_lang_sample] += 1

n_samples = 200
accuracy = {k: (v / n_samples * 100) for k, v in accuracy.items()}
pred_lang = {k: (v / n_samples * 100) for k, v in pred_lang.items()}

print("Accuracy:", dict(accuracy))
print("Pred lang:", dict(pred_lang))
