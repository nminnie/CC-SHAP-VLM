#!/usr/bin/env python3

import time, sys
import json
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
from detect_language import load_lang_detector, detect_lang

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
LANG = sys.argv[6]

model, tokenizer = load_models(model_name)
lang_detector = load_lang_detector()

if __name__ == '__main__':
    res_dict = {}
    sample_ids = []
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

            sample_ids.append(foil_id)
            formatted_samples.append(formatted_sample)
            correct_answers.append(correct_answer)
            wrong_answers.append(wrong_answer)
            image_paths.append(test_img_path)

            count += 1

    print("Done preparing data. Running test...")
    for k, (sample_id, formatted_sample, correct_answer, image_path) in enumerate(tqdm(zip(sample_ids, formatted_samples, correct_answers, image_paths), total=num_samples)):
        raw_image = Image.open(image_path) # read image
        if c_task in MULT_CHOICE_DATA.keys():
            labels = LABELS['binary']
        elif c_task in OPEN_ENDED_DATA.keys():
            labels = None
        else:
            labels = LABELS[c_task]

        inp_ask_for_prediction = prompt_answer_with_input(formatted_sample, c_task, LANG)
        print("Prompt:", inp_ask_for_prediction)
        print("Correct answer:", correct_answer)

        # Generate model response and compute MM-SHAP scores
        shap_values_prediction, mm_score_sample, num_image_patches, num_text_tokens, input_ids, output_ids = mm_shap_measure(inp_ask_for_prediction, raw_image, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS, tuple_shap_values_prediction=None)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Prediction:", prediction)

        # Evaluate model response
        prediction_lang = detect_lang(lang_detector, prediction)
        print("Pred language:", prediction_lang)
        accuracy_sample = evaluate_prediction(prediction, correct_answer, c_task)
        print("Accuracy:", accuracy_sample)
        accuracy += accuracy_sample

        mm_score += mm_score_sample
        print("MM score:", mm_score_sample)
        print("Num image patches:", num_image_patches)
        print("Num text tokens:", num_text_tokens)
        # print("SHAP values:", shap_values_prediction.values.shape)

        res_dict[f"{c_task}_{model_name}_{LANG}_{k}"] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "question": formatted_sample,
            "prompt": inp_ask_for_prediction,
            "correct_answer": correct_answer,
            "prediction": prediction,
            "prediction_lang": prediction_lang,
            "translated_prediction": "",
            "accuracy": accuracy_sample,
            "mm_score": mm_score_sample,
            "num_image_patches": num_image_patches,
            "num_text_tokens": num_text_tokens,
            "input_ids": input_ids[0].tolist(),
            "output_ids": output_ids[0].tolist(),
            "shap_values": []
            # "shap_values": shap_values_prediction.values.tolist()
        }

    save_dir = "results"
    if save_json:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/{c_task}_{model_name}_{LANG}_{count}.json", 'w') as file:
            json.dump(res_dict, file)

    print(f"Ran {TESTS} on {c_task}-{LANG} {count} samples with model {model_name}. Reporting results.\n")
    print(f"Accuracy %                       : {accuracy*100/count:.2f}  ")
    print(f"T-SHAP mean score %              : {mm_score/count*100:.2f}  ")

    c = time.time()-t1
    print(f"\nThis script ran for {c // 86400:.2f} days, {c // 3600 % 24:.2f} hours, {c // 60 % 60:.2f} minutes, {c % 60:.2f} seconds.")
