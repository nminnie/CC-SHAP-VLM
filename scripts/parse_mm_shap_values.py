import json
import numpy as np

with open("results/xgqa_bakllava_en_5.json", "r") as f:
    results = json.load(f)

for sample_id, result in results.items():
    input_ids = np.array(result["input_ids"])
    img_index = np.where(input_ids == 32000)[0][0]
    output_ids = np.array(result["output_ids"])
    shap_values = np.array(result["shap_values"])
    num_img_patches = result["num_image_patches"]
    print("\n")
    print(sample_id)
    print("image token:", img_index)
    print("patches:" , num_img_patches ** 2, "text_tokens:", len(input_ids))
    print("total_input_tokens:", num_img_patches ** 2 + len(input_ids))
    print("output:", output_ids.shape)
    print(shap_values.shape)
    arr = shap_values[0] # Convert to numpy array
    first_column = arr[:, 0]  # Extract first column
    min_value_np = np.min(first_column)
    max_value_np = np.max(first_column)
    print(f"Min: {min_value_np}, Max: {max_value_np}")

    min_index_np = np.argmin(first_column)
    max_index_np = np.argmax(first_column)
    print(f"argmin: {min_index_np}, argmax: {max_index_np}")
    