import os
import re
import json
import time
import random
import argparse

from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

import google.generativeai as genai
import google.api_core.exceptions as google_exceptions


###############################################################################
# Setup
###############################################################################

random.seed(42)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please add it to your .env file."
    )

genai.configure(api_key=GOOGLE_API_KEY)


###############################################################################
# Prompt Builder
###############################################################################

def build_prompt(scene_descriptions, language, level):

    scenes_text = "\n".join(
        [f"Scene {i+1}: {scene}"
         for i, scene in enumerate(scene_descriptions)]
    )

    prompt_v1 = f"""
You are an expert in evaluating Cultural Appropriateness across different Languages and Cultures.

You are given an image depicting a sequence of {language} story scenes.

{language} Story Scene Descriptions:

{scenes_text}

Your task is to judge the Cultural Appropriateness of the given story.

What is Cultural Appropriateness?

It measures how accurately the generated visuals represent the cultural styles, symbols, and contexts described in the scene captions across the entire story.

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

    prompt_v3 = f"""
You are an expert in evaluating Cultural Appropriateness across different Languages and Cultures.

You are given an image depicting a sequence of {language} story scenes.

{language} Story Scene Descriptions:

{scenes_text}

Your task is to judge the Cultural Appropriateness of the given story.

What is Cultural Appropriateness?

It measures how accurately the generated visuals represent the cultural styles, symbols, and contexts described in the scene captions across the entire story.

Evaluate using the following cultural dimensions:

1. Characters Facial Features
   - Assess whether facial structures align with the diversity commonly observed in {language} cultural contexts.

2. Background Objects
   - Assess furniture, decorations, tools, nearby objects and contextual elements.

3. Infrastructure
   - Assess architecture, buildings and environmental structures.

4. Apparel
   - Assess whether clothing reflects traditional or contemporary styles appropriate for {language} cultural settings.

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

    prompts = {
        1: prompt_v1,
        2: prompt_v2,
        3: prompt_v3
    }

    return prompts[level]


###############################################################################
# Gemini Call
###############################################################################

def generate_response(model, image, prompt):

    max_retries = 5
    base_delay = 10

    for retry in range(max_retries):

        try:

            response = model.generate_content(
                [image, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2048
                )
            )

            return response.text

        except google_exceptions.ResourceExhausted:

            wait_time = (
                base_delay * (2 ** retry)
                + random.uniform(0, 5)
            )

            print(
                f"Rate limit reached. "
                f"Sleeping {wait_time:.2f} sec..."
            )

            time.sleep(wait_time)

        except Exception as e:

            print(f"Error: {e}")
            return None

    return None


###############################################################################
# JSON Parser
###############################################################################

def parse_response(response_text):

    if response_text is None:
        return {
            "score": None,
            "reasoning": "Generation failed"
        }

    score = None

    match = re.search(r"\(([1-5])\)", response_text)

    if match:
        score = int(match.group(1))

    reasoning = response_text

    if "Reasoning:" in response_text:
        reasoning = response_text.split(
            "Reasoning:", 1
        )[1].strip()

    return {
        "score": score,
        "reasoning": reasoning
    }


###############################################################################
# Resume Helper
###############################################################################

def get_completed_ids(output_file):

    completed = set()

    if not os.path.exists(output_file):
        return completed

    with open(output_file, "r", encoding="utf-8") as f:

        for line in f:

            try:
                item = json.loads(line)
                completed.add(item["id"])

            except:
                pass

    return completed


###############################################################################
# Evaluation
###############################################################################

def evaluate(args):

    model = genai.GenerativeModel(args.gemini_model)

    with open(args.story_file, "r", encoding="utf-8") as f:
        story_data = json.load(f)

    completed_ids = get_completed_ids(args.output)

    print(f"Loaded {len(story_data)} stories")
    print(f"Found {len(completed_ids)} completed evaluations")

    with open(args.output, "a", encoding="utf-8") as outfile:

        for idx, sample in tqdm(
            enumerate(story_data),
            total=len(story_data)
        ):

            sample_id = idx + 1

            if sample_id in completed_ids:
                continue

            image_path = os.path.join(
                args.image_dir,
                f"{sample_id}_combine.png"
            )

            if not os.path.exists(image_path):

                print(
                    f"Missing image: {image_path}"
                )

                continue

            image = Image.open(image_path)

            scene_descriptions = [
                sample["description"]
            ]

            for item in sample["followings"]:
                scene_descriptions.append(
                    item["description"]
                )

            prompt = build_prompt(
                scene_descriptions,
                args.language,
                args.level
            )

            response_text = generate_response(
                model,
                image,
                prompt
            )

            result = parse_response(
                response_text
            )

            output_item = {
                "id": sample_id,
                "score": result["score"],
                "reasoning": result["reasoning"]
            }

            outfile.write(
                json.dumps(
                    output_item,
                    ensure_ascii=False
                )
                + "\n"
            )

            outfile.flush()

            time.sleep(2)


###############################################################################
# Main
###############################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True
    )

    parser.add_argument(
        "--language",
        required=True
    )

    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        required=True
    )

    parser.add_argument(
        "--image_dir",
        required=True
    )

    parser.add_argument(
        "--story_file",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--gemini_model",
        default="gemini-2.0-flash-001"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    evaluate(args)
