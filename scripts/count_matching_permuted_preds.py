import json
import sys

lang = sys.argv[1]

# File paths
baseline_path = f"results_perceptual_compare/xgqa_pangea_{lang}_compare_589.json"
img_path = f"results_perceptual_compare/xgqa_pangea_{lang}_compare_589_img_rnd.json"
text_path = f"results_perceptual_compare/xgqa_pangea_{lang}_compare_589_text_rnd.json"

# Load results with no permutation
with open(baseline_path, 'r') as f:
    baseline_data = json.load(f)

# Load results with permuted images
with open(img_path, 'r') as f:
    img_data = json.load(f)

# Load results with permuted texts
with open(text_path, 'r') as f:
    text_data = json.load(f)

# Function to count matching predictions
def count_matches(baseline, comparison):
    match_count = 0
    total_count = 0

    for key in baseline:
        if key not in comparison:
            continue
        pred_base = baseline[key].get("translated_prediction", "").strip().lower()
        pred_other = comparison[key].get("translated_prediction", "").strip().lower()
        if pred_base == pred_other:
            match_count += 1
        else: 
            print("Base:", pred_base, "Permuted:", pred_other, "Answer:", baseline[key]["correct_answer"], "BaseAcc:", baseline[key]["accuracy"], "PermutedAcc:", comparison[key].get("accuracy"))
        total_count += 1

    return match_count, total_count

# Compare baseline to image and text
match_img, total_img = count_matches(baseline_data, img_data)
match_text, total_text = count_matches(baseline_data, text_data)

# Report counts of matching cases
print(f"Both vs Image-Rnd: {match_img}/{total_img} predictions match")
print(f"Both vs Text-Rnd: {match_text}/{total_text} predictions match")
