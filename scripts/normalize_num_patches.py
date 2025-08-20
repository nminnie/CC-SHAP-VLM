import sys
import json
import numpy as np

lang = sys.argv[1]
results_file = f"results_num_patches/xgqa_pangea_{lang}_verify_20_2p.json"

with open(results_file, "r") as f:
    results = json.load(f)

t_shap_scores = []

for sample_id, result in results.items():
    p = int(result["num_image_patches"])
    shap_values = np.array(result["shap_values"])
    img_length = p ** 2
    text_length = shap_values.shape[1] - img_length

    image_contrib = np.abs(shap_values[0, :img_length, :]).sum() / img_length
    text_contrib = np.abs(shap_values[0, img_length:, :]).sum() / text_length
    print(image_contrib, text_contrib)
    text_score = text_contrib / (text_contrib + image_contrib)

    t_shap_scores.append(text_score)

t_shap_avg = sum(t_shap_scores) / len(t_shap_scores) * 100
print(f"T-SHAP = {t_shap_avg:.2f}")
