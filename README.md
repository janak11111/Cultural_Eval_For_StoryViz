# 🚀 A Progressive Evaluation Framework for Multicultural Analysis of Story Visualization

---

## 📌 Overview

Story visualization models generate sequences of images from textual narratives. Recent advancements in text-to-image generative models have improved narrative consistency in story visualization. However, current story visualization models often overlook cultural dimensions, resulting in visuals that lack cultural fidelity.

This repository introduces a **Progressive Cultural Evaluation Framework** for systematically assessing cultural fidelity in story visualization across multiple languages and datasets. The framework evaluates generated stories using progressively detailed cultural criteria using MLLM-as-Judge.

---

## 🧠 Framework Overview

The proposed framework consists of three major stages:

1. **Story Translation**

   * English → Hindi
   * English → Chinese

2. **Story Visualization**

   * Multilingual text-to-image generation

3. **Progressive Cultural Evaluation**

   * MLLM-as-Jury Culture Evaluation Framework
   
<p align="center">
  <img src="framework.png" width="900">
</p>

---

## 🛠️ Project Structure

```text
Cultural_Eval_For_StoryViz/
├── data/
│   ├── vist/
│   └── flintstones/
├── outputs/
│   └── images/
├── scripts/
│   ├── translate_story.py
│   ├── story_visualization.py
│   ├── culture_evaluation_with_MLLM_as_Judge_Gemini.py
│   ├── culture_evaluation_with_MLLM_as_Judge_Maya.py
│   ├── culture_evaluation_with_MLLM_as_Judge_Qwen.py
|   ├── culture_evaluation_with_MLLM_as_Jury.py
├── prompts/
├── framework.png
└── README.md
```
---

## 📂 Datasets

We evaluate our framework on two benchmark story visualization datasets.

### 1. VIST (Visual Storytelling Dataset)

* Real-world stories collected from Flickr albums
* Rich cultural diversity and real-world scenarios

### 2. FlintstonesSV

* Animated story visualization dataset based on *The Flintstones* American sitcom.
* American culture and a simpler repetitive setting

---

## 🌍 Story Translation

English narratives are translated into Hindi and Chinese using the **NLLB-200** multilingual translation model.

### Run Translation

```bash
python scripts/translate_story.py \
    --input Data/VIST/VIST_English_500.json \
    --output Data/VIST/ \
    --languages hi zh \
    --model facebook/nllb-200-distilled-600M
```
---

## 🎨 Story Visualization

Images are generated independently for each story scene using multilingual text-to-image models.

### Models Used for Story Visualization

* MuLan-SD1.5
* MuLan-SD2.1
* MuLan-SDXL

### Run Story Visualization

```bash
python Scripts/story_visualization.py \
    --input Data/VIST/VIST_English_500.json \
    --output Images/ \
    --model sdxl \
    --steps 50 \
    --guidance 7.5
```

> Each scene is generated independently due to the unavailability of multilingual story visualization models. Our primary objective is to analyze cultural fidelity rather than cross-scene consistency.
---

## ⚖️ MLLM-as-Jury Evaluation

The cultural appropriateness of each story is assessed using multiple multimodal judge models:

* Gemini Pro
* Qwen2.5-VL
* Maya

Each judge produces a score:

```text
r₁, r₂, r₃ ∈ {1, 2, 3, 4, 5}
```

The final score is computed with the aggregation function average:

```text
R = (r₁ + r₂ + r₃) / 3
```

Using multiple judges helps reduce individual model bias and improves evaluation robustness.

### Run Individual MLLM-as-Judge Evaluation

#### Gemini

```bash
python Scripts/culture_evaluation_with_MLLM_as_Judge_Gemini.py \
    --dataset VIST \
    --language English \
    --level 3 \
    --image_dir Outputs/VIST/images_english \
    --story_file Data/VIST/VIST_English_500.json \
    --output results/VIST/VIST_Gemini_English_Level3.jsonl
```

#### Qwen2.5-VL-7B-Instruct

> Set up the Qwen2.5-VL environment from Hugging Face (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) to run this script.

```bash
python Scripts/culture_evaluation_with_MLLM_as_Judge_Qwen.py \
    --dataset FlintstonesSV \
    --language Chinese \
    --level 2 \
    --image_dir Outputs/FlintstonesSV/images_chinese \
    --story_file Data/FlintstonesSV/FlintstonesSV_Chinese_500.json \
    --output results/FlintstonesSV/FlintstonesSV_Gemini_Chinese_Level2.jsonl
```

#### Maya

> Set up the Maya model environment by cloning the Maya GitHub repository (https://github.com/nahidalam/maya) and then run this script.

```bash
python Scripts/culture_evaluation_with_MLLM_as_Judge_Maya.py \
    --dataset VIST \
    --language Hindi \
    --level 1 \
    --image_dir Outputs/VIST/images_hindi \
    --story_file Data/VIST/VIST_Hindi_500.json \
    --output results/VIST/VIST_Gemini_Hindi_Level1.jsonl
```

### Agreegate MLLM-as-Jury Evalaution 

```bash
python scripts/evaluate.py \
    --images outputs/images/ \
    --stories data/vist/multilingual.json \
    --level 3 \
    --judges gemini qwen maya \
    --output results/vist_scores.json
```

---

## 🧾 Evaluation Prompts

The framework uses expert-role prompting to assess cultural appropriateness.

Prompts are available in the folder `Prompts/` directory.

--

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{kapuriya2025progressive,
  title={A Progressive Evaluation Framework for Multicultural Analysis of Story Visualization},
  author={Kapuriya, Janak and Hatami, Ali and Buitelaar, Paul},
  journal={arXiv preprint arXiv:2511.22576},
  year={2025}
}
```

---

## 🙌 Acknowledgements
This research is funded by Research Ireland under Grant Number SFI/12/RC/2289_P2 (Insight), co-funded by the European Regional Development Fund.

This work was conducted at the **Data Science Institute, University of Galway, Ireland.**

We thank the open-source communities behind NLLB, MuLan, Gemini, Qwen-VL, and other foundational models that made this research possible.
