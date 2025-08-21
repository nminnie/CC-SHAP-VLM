import sys
import json
from collections import defaultdict

lang = sys.argv[1]

# Load predictions file
results_file = f"results/pangea_1000/xgqa_pangea_{lang}_1000_translated.json"
if lang == "en":
    results_file = results_file.replace("_translated", "")
with open(results_file, "r") as f:
    results = json.load(f)

# Initialize dictionaries
lang_count = 0
accuracy = defaultdict(int)
accuracy_given_lang = defaultdict(int)
pred_lang = defaultdict(int)

# Compile accuracy and detected languages
for k, result in results.items():
    accuracy_sample = result["accuracy"]
    accuracy[accuracy_sample] += 1

    pred_lang_sample = result["prediction_lang"]
    if result["prediction"].strip().lower() == "no":
        pred_lang_sample = "en"
    
    pred_lang[pred_lang_sample] += 1

    if pred_lang_sample == lang:
        lang_count += 1
        accuracy_given_lang[accuracy_sample] += 1

# Report accuracy and detect languages for given language
n_samples = len(results)
pred_lang = {k: (v / n_samples * 100) for k, v in pred_lang.items()}
accuracy = {k: (v / n_samples * 100) for k, v in accuracy.items()}
accuracy_given_lang = {k: (v / lang_count * 100) for k, v in accuracy_given_lang.items()}

print("Pred lang:", dict(pred_lang))
print("Accuracy:", dict(accuracy))
print(f"Accuracy for {lang}:", dict(accuracy_given_lang))
