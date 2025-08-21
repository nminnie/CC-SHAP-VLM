import json
import sys

lang = sys.argv[1]

# Load the results file
with open(f'results/xgqa_llava_onevision_{lang}_choose_20_translated.json', 'r') as f:
    data = json.load(f)

mm_scores_correct = []
mm_scores_incorrect = []

# Iterate over all items
for entry in data.values():
    if entry["accuracy"] == 1:
        mm_scores_correct.append(entry["mm_score"])
    elif entry["accuracy"] == 0:
        mm_scores_incorrect.append(entry["mm_score"])

# Compute averages
average_mm_score_1 = sum(mm_scores_correct) / len(mm_scores_correct) if mm_scores_correct else 0
average_mm_score_0 = sum(mm_scores_incorrect) / len(mm_scores_incorrect) if mm_scores_incorrect else 0

# Print results
print(f"Accuracy=1. Count={len(mm_scores_correct)}. MM-SHAP={average_mm_score_1 * 100:.2f}")
print(f"Accuracy=0. Count={len(mm_scores_incorrect)}. MM-SHAP={average_mm_score_0 * 100:.2f}")