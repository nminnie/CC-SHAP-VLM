import sys
import json
import numpy as np

lang = sys.argv[1]
results_file = f"results_mm/xgqa_pangea_{lang}_query_20_translated.json"

t_shap_scores = []

with open(results_file, "r") as f:
    results = json.load(f)

# Extract MM-SHAP scores corresponding to the first output token
for sample_id, result in results.items():
    shap_values = np.array(result["shap_values"])
    p = int(result["num_image_patches"])
    img_length = p**2

    image_contrib = np.abs(shap_values[0, :img_length, 0]).sum()
    text_contrib = np.abs(shap_values[0, img_length:, 0]).sum()
    text_score = text_contrib / (text_contrib + image_contrib)
    t_shap_scores.append(text_score)

t_shap_avg = sum(t_shap_scores) / len(t_shap_scores) * 100
print(f"T-SHAP={t_shap_avg:.2f}")