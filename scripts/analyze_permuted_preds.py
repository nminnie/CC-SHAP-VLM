import json
import sys
from collections import Counter

lang = sys.argv[1]
q_type = sys.argv[2]
model = sys.argv[3]

# Specify file path according to language, question type, and model
if q_type == "compare":
    n = 589
else:
    n = 1000

if model == "pangea":
    base_file = f"results_perceptual_{q_type}/xgqa_pangea_{lang}_{q_type}_{n}_translated.json"
    img_rnd_file = f"results_perceptual_{q_type}/xgqa_pangea_{lang}_{q_type}_{n}_img_rnd_translated.json"
    text_rnd_file = f"results_perceptual_{q_type}/xgqa_pangea_{lang}_{q_type}_{n}_text_rnd_translated.json"
elif model == "llava_onevision":
    base_file = f"results_ov_perceptual_{q_type}/xgqa_llava_onevision_{lang}_{q_type}_{n}_translated.json"
    img_rnd_file = f"results_ov_perceptual_{q_type}/xgqa_llava_onevision_{lang}_{q_type}_{n}_img_rnd_translated.json"
    text_rnd_file = f"results_ov_perceptual_{q_type}/xgqa_llava_onevision_{lang}_{q_type}_{n}_text_rnd_translated.json"

if lang == "en":
    base_file = base_file.replace("_translated", "")
    img_rnd_file = img_rnd_file.replace("_translated", "")
    text_rnd_file = text_rnd_file.replace("_translated", "")

def count_preds(file_name):
    with open(file_name, "r") as f:
        results = json.load(f)

    preds = []
    answers = []
    for sample_id, result in results.items():
        pred = result["translated_prediction"] if result["translated_prediction"] != "" else result["prediction"]
        pred = pred.lower().rstrip(".")
        if q_type == "verify" or q_type == "logical":
            if "yes" in pred:
                pred = "yes"
            elif "no" in pred:
                pred = "no"
            else:
                pred = "other"

        elif q_type == "choose":
            if "right" in pred:
                pred = "right"
            elif "left" in pred:
                pred = "left"
            else:
                continue
        
        elif q_type == "query":
            if "woman" in pred or "women" in pred:
                pred = "woman"
            elif "man" in pred or "men" in pred:
                pred = "man"
            else:
                continue

        preds.append(pred)
        answers.append(result["correct_answer"])

    return dict(Counter(preds)), dict(Counter(answers))

# Count predictions when permuted, based on language and question type
count_base, count_answers = count_preds(base_file)
count_img_rnd, _ = count_preds(img_rnd_file)
count_text_rnd, _ = count_preds(text_rnd_file)

count_answers = dict(sorted(count_answers.items(), key=lambda item: item[1], reverse=True))
count_base = dict(sorted(count_base.items(), key=lambda item: item[1], reverse=True))
count_img_rnd = dict(sorted(count_img_rnd.items(), key=lambda item: item[1], reverse=True))
count_text_rnd = dict(sorted(count_text_rnd.items(), key=lambda item: item[1], reverse=True))

# Report unique predictions with permutations
print(f"\nAnswers: {count_answers}, count: {sum(count_answers.values())}")
print(f"\nBase: {count_base}")
print(f"\nImg rnd: {count_img_rnd}")
print(f"\nText rnd: {count_text_rnd}")