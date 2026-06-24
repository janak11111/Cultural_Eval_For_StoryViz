import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# NLLB language codes
LANG_CODES = {
    "hi": "hin_Deva",   # Hindi
    "zh": "zho_Hans"    # Chinese Simplified
}


def load_model(model_name):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Using device: {device}")
    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, target_lang):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=512
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def process_data(data, tokenizer, model, device, target_lang):
    for item in tqdm(data, desc=f"Translating → {target_lang}"):

        # Collect all text (main + followings)
        texts = [item["description"]]

        for f in item.get("followings", []):
            texts.append(f["description"])

        # Translate in one batch
        translations = translate_batch(
            texts, tokenizer, model, device, target_lang
        )

        # Assign back
        item["description"] = translations[0]

        for i, f in enumerate(item.get("followings", [])):
            f["description"] = translations[i + 1]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Target languages (hi zh)"
    )
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="NLLB model name"
    )

    args = parser.parse_args()

    # Extract dataset name from input path
    dataset = os.path.basename(args.input)          # VIST_English_500.json
    dataset = os.path.splitext(dataset)[0]          # VIST_English_500

    print(f"Dataset detected: {dataset}")

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load model
    tokenizer, model, device = load_model(args.model)

    os.makedirs(args.output, exist_ok=True)

    # Translate per language
    for lang in args.languages:

        if lang not in LANG_CODES:
            print(f" Skipping unsupported language: {lang}")
            continue

        lang_code = LANG_CODES[lang]

        print(f"\n Translating to {lang} ({lang_code})")

        import copy
        translated_data = process_data(
            copy.deepcopy(data),
            tokenizer,
            model,
            device,
            lang_code
        )

        # Save using dataset name
        output_file = os.path.join(
            args.output,
            f"{dataset}_translated_{lang}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
