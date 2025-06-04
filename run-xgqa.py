import time, sys
import torch
print("Cuda is available:", torch.cuda.is_available())
from accelerate import Accelerator
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import random, os
from tqdm import tqdm

from load_models import load_models
from read_datasets import read_data
from generation_and_prompting import *
from mm_shap_cc_shap import *
from config import *

torch.cuda.empty_cache()

from transformers.utils import logging
logging.set_verbosity_error()
import logging
logging.getLogger('shap').setLevel(logging.ERROR)

random.seed(42)

t1 = time.time()

c_task = sys.argv[1]
model_name = sys.argv[2]
num_samples = int(sys.argv[3])
save_json = int(sys.argv[4])
data_root = sys.argv[5]

model, tokenizer = load_models(model_name)


if __name__ == '__main__':
    formatted_samples, correct_answers, wrong_answers, image_paths = [], [], [], []
    accuracy = 0
    count = 0
    mm_score = 0

    print("Preparing data...")
    if c_task in MULT_CHOICE_DATA.keys():
        pass

    elif c_task in OPEN_ENDED_DATA.keys(): # open ended generation tasks
        images_path = f"{data_root}{OPEN_ENDED_DATA[c_task][0]}"
        qa_path = f"{data_root}{OPEN_ENDED_DATA[c_task][1]}"
        vqa_data = read_data(c_task, qa_path, images_path, data_root)
        for foil_id, foil in tqdm(vqa_data.items()):  # tqdm
            if count + 1 > num_samples:
                break
            test_img_path = os.path.join(images_path, foil["image_file"])
            question = foil["caption"]
            formatted_sample = format_example_vqa_gqa(question) # takes in question
            if c_task == 'vqa':
                correct_answer = foil["answers"] # there are multiple answers annotations
            else:
                correct_answer = foil["answer"]
            wrong_answer = "impossible to give"

            formatted_samples.append(formatted_sample)
            correct_answers.append(correct_answer)
            wrong_answers.append(wrong_answer)
            image_paths.append(test_img_path)

            count += 1

    print("Done preparing data. Running test...")

    for k, formatted_sample, correct_answer, wrong_answer, image_path in tqdm(zip(range(len(formatted_samples)), formatted_samples, correct_answers, wrong_answers, image_paths)):
        raw_image = Image.open(image_path) # read image
        if c_task in MULT_CHOICE_DATA.keys():
            labels = LABELS['binary']
        elif c_task in OPEN_ENDED_DATA.keys():
            labels = None
        else:
            labels = LABELS[c_task]

        inp_ask_for_prediction = prompt_answer_with_input(formatted_sample, c_task)
        prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=labels)
        accuracy_sample = evaluate_prediction(prediction, correct_answer, c_task)
        accuracy += accuracy_sample

        mm_score_sample, tuple_shap_values_prediction = mm_shap_measure(inp_ask_for_prediction, raw_image, model, tokenizer, max_new_tokens=5, tuple_shap_values_prediction=None)
        mm_score += mm_score_sample

    print(f"Ran {TESTS} on {c_task} {count} samples with model {model_name}. Reporting results.\n")
    print(f"Accuracy %                       : {accuracy*100/count:.2f}  ")
    print(f"T-SHAP mean score %              : {mm_score/count*100:.2f}  ")

    c = time.time()-t1
    print(f"\nThis script ran for {c // 86400:.2f} days, {c // 3600 % 24:.2f} hours, {c // 60 % 60:.2f} minutes, {c % 60:.2f} seconds.")