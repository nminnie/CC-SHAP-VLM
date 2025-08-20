import json
import sys

lang = sys.argv[1]

# Load the JSON file
with open(f'results/xgqa_llava_onevision_{lang}_choose_20_translated.json', 'r') as f:
    data = json.load(f)

# Initialize accumulators
mm_scores_accuracy_1 = []
mm_scores_accuracy_0 = []

# Iterate over all entries
for entry in data.values():
    if entry["accuracy"] == 1:
        mm_scores_accuracy_1.append(entry["mm_score"])
    elif entry["accuracy"] == 0:
        mm_scores_accuracy_0.append(entry["mm_score"])

# Compute averages
average_mm_score_1 = sum(mm_scores_accuracy_1) / len(mm_scores_accuracy_1) if mm_scores_accuracy_1 else 0
average_mm_score_0 = sum(mm_scores_accuracy_0) / len(mm_scores_accuracy_0) if mm_scores_accuracy_0 else 0

# Print results
print(f"Accuracy=1. Count={len(mm_scores_accuracy_1)}. MM-SHAP={average_mm_score_1 * 100:.2f}")
print(f"Accuracy=0. Count={len(mm_scores_accuracy_0)}. MM-SHAP={average_mm_score_0 * 100:.2f}")