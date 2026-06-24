import argparse
import json
import os
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
import mulankit

# Load pipeline + correct MuLan adapter
def load_pipeline(model_name):
    print(f"Loading model: {model_name}")

    if model_name == "sdxl":
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        adapter = "mulanai/mulan-lang-adapter::sdxl_aesthetic.pth"
        variant = "fp16"
        use_safetensors = True

    elif model_name == "sd21":
        base_model = "stabilityai/stable-diffusion-2-1"
        adapter = "mulanai/mulan-lang-adapter::sd21_aesthetic.pth"
        variant = None
        use_safetensors = False

    elif model_name == "sd15":  # dreamshaper
        base_model = "Lykon/dreamshaper-8"
        adapter = "mulanai/mulan-lang-adapter::sd15_aesthetic.pth"
        variant = None
        use_safetensors = False

    else:
        raise ValueError("model must be: sdxl | sd21 | sd15")

    # Load base model
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant=variant if variant else None,
        use_safetensors=use_safetensors
    )

    # Apply MuLan adapter
    mulankit.setup(
        force_sdxl_zero_empty_prompt=False,
        force_sdxl_zero_pool_prompt=False
    )

    pipe = mulankit.transform(
        pipe,
        adapter_path=adapter
    )

    pipe = pipe.to("cuda", dtype=torch.float16)

    print(f"Loaded {model_name} with adapter")
    return pipe


# Parse dataset + language
def parse_filename(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]

    if "_translated_" in name:
        base, lang = name.split("_translated_")
    else:
        base, lang = name, "en"

    return base, lang


# Load prompts
def load_stories(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stories = []

    for sample in data:
        stories.append((sample["description"], sample["id"], sample["image_id"]))

        for s in sample.get("followings", []):
            stories.append((s["description"], sample["id"], s["image_id"]))

    return stories


# Generate images
def generate(pipe, stories, output_dir, steps, guidance):
    os.makedirs(output_dir, exist_ok=True)

    for prompt, sid, img_id in tqdm(stories, desc="Generating"):
        try:

            generator = torch.Generator(device=pipe.device).manual_seed(12345)
            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
                )
            ).images[0]

            save_path = os.path.join(output_dir, f"{sid}_{img_id}.jpg")
            image.save(save_path)

        except Exception as e:
            print(f"Failed {sid}_{img_id}: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="JSON file or folder")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--model",
        required=True,
        choices=["sdxl", "sd21", "sd15"]
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)

    args = parser.parse_args()

    pipe = load_pipeline(args.model)

    # collect input files
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".json")
        ]

    # process each dataset-language separately
    for file_path in files:
        dataset, lang = parse_filename(file_path)

        output_dir = os.path.join(
            args.output,
            args.model,        # ✅ model-wise
            dataset,           # ✅ dataset-wise
            lang               # ✅ language-wise
        )

        print(f"\n📂 Dataset: {dataset}")
        print(f"🌍 Language: {lang}")
        print(f"🤖 Model: {args.model}")

        stories = load_stories(file_path)

        generate(pipe, stories, output_dir, args.steps, args.guidance)


if __name__ == "__main__":
    main()
``
