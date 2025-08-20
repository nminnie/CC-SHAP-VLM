import json
import sys

lang = sys.argv[1]

# Load the JSON file
with open(f'results_mm/xgqa_pangea_{lang}_query_20.json', 'r') as f:
    data = json.load(f)

# Initialize accumulators
num_patches = []
mm_scores_accuracy_0 = []

# Iterate over all entries
for entry in data.values():
    num_patches.append(entry["num_image_patches"])

print(num_patches)
# Compute averages
average_p = sum(num_patches) / len(num_patches) 

# Print results
print(f"Lang={lang}. p={average_p:.2f}")
