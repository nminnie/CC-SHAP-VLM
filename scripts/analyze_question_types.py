import json
import sys
from collections import defaultdict

with open("data/xGQA/testdev_balanced_questions.json", "r") as f:
    data = json.load(f)

lang = sys.argv[1]
filename = f"results/pangea_1000_v2/xgqa_pangea_{lang}_1000_translated.json"
if lang == "en":
    filename = filename.replace("_translated", "")
with open(filename, "r") as f:
    results = json.load(f)

compiled_results = {}
structural_types = defaultdict(int)
dimension = "semantic"
max_steps = 5

for key, result in results.items():
    sample_id = result["sample_id"]
    result.pop('sample_id', None)
    result.pop('prompt', None)
    result.pop('input_ids', None)
    result.pop('output_ids', None)
    result["structural"] = data[sample_id]["types"]["structural"]
    result["semantic"] = data[sample_id]["types"]["semantic"]
    result["reasoning_steps"] = len(data[sample_id]["semantic"])

    compiled_results[sample_id] = result
    
    if dimension == "reasoning_steps" and result[dimension] >= max_steps:
        structural_types[max_steps] += 1
    else:
        structural_types[result[dimension]] += 1


correct_types = defaultdict(int)
incorrect_types = defaultdict(int)
for key, result in compiled_results.items():
    if result["accuracy"] == 1:
        correct_types[result[dimension]] += 1
    else:
        incorrect_types[result[dimension]] += 1

for dim_type, count in structural_types.items():
    correct_perc = round(correct_types[dim_type] / count * 100, 1)
    incorrect_perc = round(incorrect_types[dim_type] / count * 100, 1)
    print(f"{dimension}: {dim_type}. Count={count}, Correct={correct_perc}")

