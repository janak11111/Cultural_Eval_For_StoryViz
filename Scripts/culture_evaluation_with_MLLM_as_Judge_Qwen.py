import os
import json
import torch
import random
import argparse

from tqdm import tqdm
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TorchAoConfig
)

from qwen_vl_utils import process_vision_info


random.seed(42)


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)


###############################################################################
# PROMPT BUILDER 
###############################################################################

def build_prompt(scene_descriptions, language, level):

    prompt_v1 = f"""You are an expert in evaluating Cultural Appropriateness across different Languages and Cultures.

You are given an image depicting a sequence of {language} story scenes.

{language} Story Scene Descriptions are given below:

Scene 1: {scene_descriptions[0]}
Scene 2: {scene_descriptions[1]}
Scene 3: {scene_descriptions[2]}
Scene 4: {scene_descriptions[3]}
Scene 5: {scene_descriptions[4]}

Your task is to judge the Cultural Appropriateness of the given story. Evaluate the Cultural Appropriateness across the entire story.

What is Cultural Appropriateness? It measures how accurately the generated visuals represent the cultural styles, symbols, and contexts described in the scene captions across the entire story.

Rating Scale:

5 – Perfect
4 – Minor Issues
3 – Major Issues
2 – Significant Issues
1 – Rubbish

Return the Rating based on the provided rating scale and explain the Reasoning behind the given rating.

Follow the below given format to return Rating. 

single rating number (5, 4, 3, 2, or 1) enclosed in parentheses.

Reasoning:"""

    prompt_v2 = f"""
You are an expert in evaluating Cultural Appropriateness across different Languages and Cultures.

You are given an image depicting a sequence of {language} story scenes.

{language} Story Scene Descriptions:

{scenes_text}

Your task is to judge the Cultural Appropriateness of the given story.

What is Cultural Appropriateness?

It measures how accurately the generated visuals represent the cultural styles, symbols, and contexts described in the scene captions across the entire story.

Focus on:

1. Facial Features
2. Background Objects
3. Infrastructure
4. Apparel

when evaluating cultural appropriateness.

Rating Scale:

5 - Perfect
4 - Minor Issues
3 - Major Issues
2 - Significant Issues
1 - Rubbish

Return the Rating based on the provided rating scale and explain the Reasoning behind the given rating.

Follow the below given format to return Rating.

single rating number (5, 4, 3, 2, or 1) enclosed in parentheses.

Reasoning:"""

    prompt_v3 = f"""You are an expert in evaluating Cultural Appropriateness across different Languages and Cultures.

You are given an image depicting a sequence of {language} story scenes.

{language} Story Scene Descriptions are given below:

Scene 1: {scene_descriptions[0]}
Scene 2: {scene_descriptions[1]}
Scene 3: {scene_descriptions[2]}
Scene 4: {scene_descriptions[3]}
Scene 5: {scene_descriptions[4]}

Your task is to judge the Cultural Appropriateness of the given story. Evaluate the Cultural Appropriateness across the entire story.

What is Cultural Appropriateness? It measures how accurately the generated visuals represent the cultural styles, symbols, and contexts described in the scene captions across the entire story.

Focus on evaluating the character's facial structures, background objects, infrastructures, and apparel during the evaluation of {language} cultural Aspects by analyzing the story scene images as given below.

1. Characters Facial Features: Evaluate whether the facial structures align with the diverse traits commonly found in {language} culture. Avoid assumptions about stereotypical features.  
2. Background Objects: Assess whether the depicted objects represent the {language} cultural setting described in the scenes, focusing on nearby objects, furniture, decorations, and other contextual details.  
3. Infrastructures: Consider whether the settings, such as architectural elements, are appropriate for the {language} culture.  
4. Apparel: Assess whether the clothing aligns with traditional or contemporary styles representative of {language} culture.

Rating Scale:

5 – Perfect
4 – Minor Issues
3 – Major Issues
2 – Significant Issues
1 – Rubbish

Return the Rating based on the provided rating scale and explain the Reasoning behind the given rating.

Follow the below given format to return Rating. 

single rating number (5, 4, 3, 2, or 1) enclosed in parentheses.

Reasoning:"""

    prompts = {1: prompt_v1, 2: prompt_v2, 3: prompt_v3}
    return prompts[level]


def generate_solution(scene_descriptions, image_path, args):

    prompt = build_prompt(scene_descriptions, args.language, args.level)

    full_image_path = os.path.join(args.image_dir, image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": full_image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0
    )

    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return output_text[0]

def evaluate(args):

    with open(args.story_file, "r", encoding="utf-8") as f:
        story_data = json.load(f)

    with open(args.output, "w", encoding="utf-8") as ans_file:

      if args.dataset == "FlintstonesSV":

            for i, sample in tqdm(enumerate(story_data)):

                scene_descriptions = [sample["description"]]
                image_id = sample["image_id"][:-4]

                for x in sample["followings"]:
                    scene_descriptions.append(x["description"])

                try:
                    response = generate_solution(
                        scene_descriptions,
                        f"{image_id}.jpg",
                        args
                    )

                    ans_file.write(json.dumps({
                        "id": image_id,
                        "prediction": response
                    }, ensure_ascii=False) + "\n")

                    ans_file.flush()

                    print(">>>>> Done", i)

                except Exception as e:
                    print(f"Error Flintstones {i}: {e}")

        else:

            for i, sample in tqdm(enumerate(story_data)):

                scene_descriptions = [sample["description"]]

                sample_id = sample["id"]
                image_id = sample["image_id"]

                for x in sample["followings"]:
                    scene_descriptions.append(x["description"])

                try:
                    response = generate_solution(
                        scene_descriptions,
                        f"{sample_id}_{image_id}.jpg",
                        args
                    )

                    ans_file.write(json.dumps({
                        "id": image_id,
                        "prediction": response
                    }, ensure_ascii=False) + "\n")

                    ans_file.flush()

                    print(">>>>> Done", i)

                except Exception as e:
                    print(f"Error VIST {i}: {e}")

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--language", required=True)

    parser.add_argument("--level", type=int, choices=[1, 2, 3], required=True)

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--story_file", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    print("---------------------------")
    print(f"Dataset: {args.dataset}")
    print(f"Language: {args.language}")
    print(f"Level: {args.level}")
    print("---------------------------")

    evaluate(args)
