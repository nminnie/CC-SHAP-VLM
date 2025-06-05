#!/usr/bin/env python3

import time, sys
import torch
print("Cuda is available:", torch.cuda.is_available())
from PIL import Image
import random, os
from tqdm import tqdm

from load_models import load_models
from read_datasets import read_data
from generation_and_prompting import *
from mm_shap_cc_shap import mm_shap_measure
from config import *
from translate import load_translation_model, translate_text

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
nllb_model, nllb_tokenizer = load_translation_model()

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

    total_vlm_time = 0
    total_mm_shap_time = 0

    for k, (formatted_sample, correct_answer, image_path) in enumerate(tqdm(zip(formatted_samples, correct_answers, image_paths), total=num_samples)):
        raw_image = Image.open(image_path) # read image
        if c_task in MULT_CHOICE_DATA.keys():
            labels = LABELS['binary']
        elif c_task in OPEN_ENDED_DATA.keys():
            labels = None
        else:
            labels = LABELS[c_task]

        inp_ask_for_prediction = prompt_answer_with_input(formatted_sample, c_task, LANG)

        start_vlm = time.time()
        prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=labels)
        end_vlm = time.time()
        total_vlm_time += (end_vlm - start_vlm)
        print("Raw output:", prediction)

        if LANG != "en": # Translate model response
            prediction = translate_text(nllb_model, nllb_tokenizer, prediction, src_lang=LANG, tgt_lang="en")
        print("Translated output:", prediction)
        accuracy_sample = evaluate_prediction(prediction, correct_answer, c_task)
        print("Accuracy:", accuracy_sample)
        accuracy += accuracy_sample

        start_mm_time = time.time()
        # mm_score_sample, tuple_shap_values_prediction = mm_shap_measure(inp_ask_for_prediction, raw_image, model, tokenizer, max_new_tokens=5, tuple_shap_values_prediction=None)
        mm_score_sample, _ = 0, 0
        end_mm_time = time.time()
        total_mm_shap_time += (end_mm_time - start_mm_time)

        mm_score += mm_score_sample

    print(f"Ran {TESTS} on {c_task} {count} samples with model {model_name}. Reporting results.\n")
    print(f"Accuracy %                       : {accuracy*100/count:.2f}  ")
    print(f"T-SHAP mean score %              : {mm_score/count*100:.2f}  ")

    c = time.time()-t1
    print(f"\nThis script ran for {c // 86400:.2f} days, {c // 3600 % 24:.2f} hours, {c // 60 % 60:.2f} minutes, {c % 60:.2f} seconds.")
    print("VLM generation time:", total_vlm_time)
    print("MM-SHAP computation time:", total_mm_shap_time)