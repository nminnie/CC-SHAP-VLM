import sys
from PIL import Image

from generation_and_prompting import prompt_answer_with_input, vlm_predict
from load_models import load_models

c_task = sys.argv[1]
model_name = sys.argv[2]
# model, tokenizer = load_models(model_name)

image_path = "./data/xGQA/testdev_balanced_images/n446242.jpg"
raw_image = Image.open(image_path)

questions = {
    "bn": "\u09aa\u09cd\u09af\u09be\u09a8\u09cd\u099f\u0997\u09c1\u09b2\u09cb\u09b0 \u09b0\u0999 \u0995\u09c0?",
    "de": "Welche Farbe hat die Hose?",
    "en": "What color are the pants?",
    "id": "Apa warna celana-celana itu ?",
    "ko": "\ubc14\uc9c0\ub294 \ubb34\uc2a8 \uc0c9\uc774\uc5d0\uc694?",
    "pt": "Qual \u00e9 a cor das cal\u00e7as?",
    "ru": "\u041a\u0430\u043a\u043e\u0433\u043e \u0446\u0432\u0435\u0442\u0430 \u0431\u0440\u044e\u043a\u0438?",
    "zh": "\u8fd9\u4e9b\u88e4\u5b50\u662f\u4ec0\u4e48\u989c\u8272\u7684\uff1f"
}

def generate_answer(question):
    inp_ask_for_prediction = prompt_answer_with_input(question, c_task)
    print(inp_ask_for_prediction)
    # prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=None)
    prediction = "red"
    return prediction

for lang, question in questions.items():
    print("\nLanguage:", lang)
    output = generate_answer(question)
    print("OUTPUT:", output)