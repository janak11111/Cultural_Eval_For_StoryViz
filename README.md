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
Eval_Multicultural_Story_Visualization/
├── data/
│   ├── vist/
│   └── flintstones/
├── outputs/
│   └── images/
├── results/
├── scripts/
│   ├── translate.py
│   ├── generate_images.py
│   ├── evaluate.py
├── prompts/
├── assets/
│   └── framework.png
└── README.md
```
---

## 📂 Datasets

We evaluate our framework on two benchmark story visualization datasets.

### 1. VIST (Visual Storytelling Dataset)

* Real-world stories collected from Flickr albums
* Approximately 50,000 stories
* Rich cultural diversity and real-life scenarios

### 2. FlintstonesSV

* Animated story visualization dataset based on *The Flintstones* american series
* Approximately 24,000 image-caption pairs
* Simplified and repetitive visual environments

---

## 🌍 Story Translation

English narratives are translated into Hindi and Chinese using the **NLLB-200** multilingual translation model.

### Run Translation

```bash
python scripts/translate.py \
    --input data/vist/en_stories.json \
    --output data/vist/multilingual.json \
    --languages hi zh \
    --model facebook/nllb-200
```
---

## 🎨 Story Visualization

Images are generated independently for each story scene using multilingual text-to-image models.

### Models Evaluated

* MuLan-SD1.5
* MuLan-SD2.1
* MuLan-SDXL

### Run Image Generation

```bash
python scripts/generate_images.py \
    --input data/vist/multilingual.json \
    --output outputs/images/ \
    --model mulan-sdxl \
    --steps 50 \
    --guidance 7.5
```

> Each scene is generated independently without temporal conditioning because of the lack of multilingual story visualization models.

---

## 📊 Progressive Cultural Evaluation

We propose a three-level evaluation framework that progressively increases cultural guidance.

### Level 1: Basic Cultural Criteria (C)

Evaluates Cultural Appropriateness based on the basic cultural criteria derived directly from the story context.

### Level 2: Cultural Dimension Guidance 

- Basic Cultural Criteria (C) + Focus Points (F)

Introduces a specific cultural dimension to pay attention to with basic cultural criteria during the evaluation.

* Background Objects
* Facial Features
* Infrastructure
* Apparel

### Level 3: Cultural Examples Grounding

- Basic Cultural Criteria (C) + Focus Points (F) +  Illustrative Examples (E) 

It provides concrete examples for each focus point to guide consistent and precise assessment.

- Background Objects: Assess whether the depicted objects represent the target culture setting described in the scenes, focusing on nearby objects, furniture, decorations and other contextual details.

– Facial Features: Evaluate whether the facial structures align with the diverse traits commonly found in the target culture. Avoid assumptions about stereotypical features.

– Infrastructures: Consider whether the settings, such as architectural elements, are appropriate for the target culture.

– Apparel: Assess whether clothing aligns with traditional or contemporary styles representative of the appropriate culture.


## ⚖️ MLLM-as-Jury Evaluation

Cultural fidelity is assessed using multiple multimodal judges:

* Gemini Pro
* Qwen2.5-VL
* Maya

Each judge produces a score:

```text
r₁, r₂, r₃ ∈ {1,2,3,4,5}
```

The final score is computed as:

```text
R = (r₁ + r₂ + r₃) / 3
```

Using multiple judges helps reduce individual model bias and improve evaluation robustness.

### Run Evaluation

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

Prompts are given in the folder /Prompts directory.

--

## 📈 Key Findings

* ✅ MuLan-SDXL consistently achieves the highest cultural appropriateness scores.
* ✅ Real-world stories (VIST) generally outperform animated stories (FlintstonesSV).
* ✅ English narratives receive higher scores than Hindi and Chinese translations.
* ✅ More detailed evaluation guidance leads to stricter and more discriminative scoring.
* ⚠️ Generated outputs occasionally contain cultural inconsistencies and stereotypical representations.
* ⚠️ Translation errors can propagate and negatively affect downstream visual generation quality.

---

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

## 🔗 Repository

GitHub Repository:

https://github.com/janak11111/Cultural_Eval_For_StoryViz

---

## 🙌 Acknowledgements

This work was conducted at the **Data Science Institute, University of Galway**.

We thank the open-source communities behind NLLB, MuLan, Gemini, Qwen-VL, and other foundational models that made this research possible.
