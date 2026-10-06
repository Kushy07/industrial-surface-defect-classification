# Industrial Surface Defect Classification using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automatic classification of **steel surface defects** using a lightweight Transfer Learning approach with **MobileNetV2**.

---

## Problem Statement

Given a steel surface image, predict the type of surface defect present in the image.

**Defect Categories (NEU Surface Defect Dataset):**

| Class              | Description                              |
|--------------------|------------------------------------------|
| Crazing            | Network of fine surface cracks           |
| Inclusion          | Foreign particles embedded in the surface|
| Patches            | Irregular textured patches               |
| Pitted Surface     | Small pits / holes                       |
| Rolled-in Scale    | Scale pressed into metal during rolling  |
| Scratches          | Linear surface marks                     |

---

## Approach

- **Model:** MobileNetV2 (pretrained on ImageNet)
- **Strategy:** Partial Transfer Learning (feature extractor frozen)
- **Head:** GlobalAveragePooling2D → Dense(6, softmax)
- **Input size:** 224 × 224 RGB
- **Training:** Only the classification head is trained

---

## Results

| Metric                    | Value     |
|---------------------------|-----------|
| Validation Accuracy       | **98.06%**|
| Validation Loss           | 0.0574    |
| Training Accuracy         | 99.86%    |
| Training Loss             | 0.0145    |
| Loss Gap (Val − Train)    | ~0.043    |

**Key Observation:**  
Almost perfect classification on 5 out of 6 classes. Remaining errors are concentrated in the `inclusion` class (visually similar to scratches / pitted surface).

---
