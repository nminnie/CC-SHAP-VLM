import json

base_file = "results_perceptual_verify/xgqa_pangea_en_verify_1000.json"
permuted_file = "results_perceptual_verify/xgqa_pangea_en_verify_1000_text_rnd.json"

with open(base_file, "r") as f:
    base_results = json.load(f)

with open(permuted_file, "r") as f:
    permuted_results = json.load(f)

for sample_id, result in permuted_results.items():
    base_result = base_results[sample_id]
    base_pred = base_result["translated_prediction"] if base_result["translated_prediction"] else base_result["prediction"]
    base_question = base_result["question"]
    base_acc = base_result["accuracy"]

    if result["accuracy"] == 1:
        correct_answer = result["correct_answer"]
        pred = result["translated_prediction"] if result["translated_prediction"] else result["prediction"]
        question = result["question"]
        print(f"\nID: {sample_id}")
        print(f"Original - Q: {base_question}. A: {base_pred}. Ground truth: {correct_answer}. Acc={base_acc}")
        print(f"Permuted - Q: {question}. A: {pred}. Acc={result['accuracy']}")
        print(result["image_path"])