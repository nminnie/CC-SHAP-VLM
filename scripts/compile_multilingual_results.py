import json
import os

langs = ["bn", "de", "en", "id", "ko", "pt", "ru", "zh"]

def load_results_file(lang):
    with open(filename.format(lang=lang), "r") as f:
        results = json.load(f)
    return results

with open("data/xGQA/testdev_balanced_questions.json", "r") as f:
    data = json.load(f)

filename = "results/pangea1000/xgqa_pangea_{lang}_1000_processed.json"

results = {lang: load_results_file(lang) for lang in langs}
results_en = results["en"]

compiled_results = {}
for sample_id, result in results_en.items():
    correct_langs = [lang for lang in langs if results[lang][sample_id]["accuracy"] == 1]
    total_acc = len(correct_langs)
    # if total_acc != 0:
    #     continue
    preds = [f"{results[lang][sample_id]['translated_prediction']} ({lang})" if results[lang][sample_id]["translated_prediction"] != "" else f"{results[lang][sample_id]['prediction']} ({lang})" for lang in langs]
    compiled_results[sample_id] = {
        "question": data[sample_id]["question"],
        "answer": data[sample_id]["answer"],
        "image_path": result["image_path"],
        "structural": data[sample_id]["types"]["structural"],
        "semantic": data[sample_id]["types"]["semantic"],
        "reasoning_steps": len(data[sample_id]["semantic"]),
        "predictions": preds,
        "correct_langs": correct_langs,
        "total_acc": total_acc
    }

with open("results/pangea1000/compiled_results.json", "w") as f:
    json.dump(compiled_results, f)