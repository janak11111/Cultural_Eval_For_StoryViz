import os
import json
import time
import random
import argparse

from tqdm import tqdm
from PIL import Image

from llava.eval.talk2maya import run_vqa_model

random.seed(42)


###############################################################################
# PROMPTS 
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


    prompt_v2 = f"""You are an expert in evaluating Cultural Appropriateness across different Languages and Cultures.

You are given an image depicting a sequence of {language} story scenes.

{language} Story Scene Descriptions are given below:

Scene 1: {scene_descriptions[0]}
Scene 2: {scene_descriptions[1]}
Scene 3: {scene_descriptions[2]}
Scene 4: {scene_descriptions[3]}
Scene 5: {scene_descriptions[4]}

Your task is to judge the Cultural Appropriateness of the given story. Evaluate the Cultural Appropriateness across the entire story.

What is Cultural Appropriateness? It measures how accurately the generated visuals represent the cultural styles, symbols, and contexts described in the scene captions across the entire story.

Focus on evaluating the character's facial structures, background objects, infrastructures, and apparel during the evaluation of {language} cultural Aspect by analyzing the story scene images.

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


###############################################################################
# MODEL CALL
###############################################################################

def generate_solution(scene_descriptions, image_path, language, level, image_dir):

    full_image_path = os.path.join(image_dir, image_path)

    prompt = build_prompt(scene_descriptions, language, level)

    _, wrapped_answer = run_vqa_model(
        question=prompt,
        image_file=full_image_path,
        temperature=0.0,
        top_p=1,
        max_new_tokens=4096
    )

    return wrapped_answer


def evaluate(args):

    with open(args.story_file, "r", encoding="utf-8") as f:
        story_data = json.load(f)

    with open(args.output, "w", encoding="utf-8") as out_f:

        # =========================
        # CASE 1: FLINTSTONES
        # =========================
        if args.dataset == "FlintstonesSV":

            all_images = os.path.join(args.image_dir, "FlintstonesSV")

            for i, sample in tqdm(enumerate(story_data[:])):

                scene_descriptions = [sample["description"]]
                image_id = sample["image_id"][:-4]

                for x in sample["followings"]:
                    scene_descriptions.append(x["description"])

                try:
                    response = generate_solution(
                        scene_descriptions,
                        f"{image_id}.jpg",
                        args.language,
                        args.level,
                        all_images
                    )

                    out_f.write(json.dumps({
                        "id": image_id,
                        "prediction": response
                    }, ensure_ascii=False) + "\n")

                    out_f.flush()

                    print(">>>>> Done", i)

                except Exception as e:
                    print(f"Error Flintstones {i}: {e}")

        # =========================
        # CASE 2: VIST 
        # =========================
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
                        args.language,
                        args.level,
                        args.image_dir
                    )

                    out_f.write(json.dumps({
                        "id": image_id,
                        "prediction": response
                    }, ensure_ascii=False) + "\n")

                    out_f.flush()

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
