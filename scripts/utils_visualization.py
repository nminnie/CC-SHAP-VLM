import json
import torch
from transformers import AutoProcessor

# Load model processor
processor = AutoProcessor.from_pretrained("neulab/Pangea-7B-hf")

# Load results file
results_file = "results_num_patches/xgqa_pangea_en_verify_20_halfp.json"
sample_id = "20381460"
with open(results_file, "r") as f:
    results = json.load(f)

# Report number of image and text tokens
result = results[sample_id]
input_ids = result["input_ids"]
print("Num image patches:", result["num_image_patches"])
print("Num text tokens:", result["num_text_tokens"])

# Tokenize input and output sequences
tokens = [processor.tokenizer.decode(torch.tensor([ids])).strip(" ") for ids in input_ids]
print(f"Input tokens: {tokens}")

output_ids = result["output_ids"]
decoded_output_tokens = [processor.tokenizer.decode(ids) for ids in output_ids]
print(f"Output tokens: {decoded_output_tokens}")

# Report the corresponding MM-SHAP scores
shap_values = result["shap_values"][0]
shap_values = [round(score[0], 2) for score in shap_values]
print(f"Scores: {shap_values}")
