# Concept Frustration: Aligning Human Concepts and Machine Representations in Interpretable AI

This repository contains the code, experiments, and selected results for the paper:

**Aligning Human Concepts and Machine Representations in Interpretable AI**  
Enrico Parisini, Christopher Soelistyo, Ahab Isaac, Alessandro Barp, Christopher R. S. Banerji

---

## Overview

Concept-based models aim to make machine learning systems interpretable by structuring predictions around human-understandable concepts. However, real-world data often contains latent structure that is not captured by predefined concept ontologies.

This repository accompanies a study investigating **concept frustration**, a phenomenon where:

- supervised concept representations  
- unsupervised machine representations  
- task structure  

cannot all be simultaneously aligned within a single concept geometry.

When this occurs, concept-based models may exhibit systematic misalignment between human concepts and machine representations.

The experiments in this repository explore this phenomenon across several settings:

- A geometric toy task (globe / treasure hunter)
- Synthetic simulations
- A natural language task (sarcasm detection)
- A computer vision task (CUB bird dataset)

These experiments illustrate how incomplete concept ontologies can lead to structural misalignment between human concepts and machine-learned representations.

---

## Repository Structure

- **notebooks/**  
  Jupyter notebooks implementing the main experiments.

- **results/**  
  Output CSV files containing experiment results and metrics.

- **figures/**  
  Figures used in the manuscript.

- **src/**  
  Optional reusable code modules.

---

## Experiments

### 1. Globe / Treasure Hunter Toy Environment

A geometric environment used to illustrate concept misalignment.

Notebook: notebooks/01_globe_treasure_hunter.ipynb

---

### 2. Synthetic Concept Geometry Simulations

Explores theoretical behaviour of concept-based models under controlled geometric structures.

Notebook: notebooks/02_synthetic_simulations.ipynb


---

### 3. Sarcasm Detection (NLP Task)

Uses language model embeddings to study concept frustration in a natural language setting.

Concepts represent interpretable linguistic attributes, while embeddings capture richer semantic structure.

Notebook: notebooks/03_sarcasm_task.ipynb


---

### 4. CUB Bird Classification

Uses vision model embeddings to study concept frustration in a computer vision setting.

Concepts represent interpretable visual attributes, while embeddings capture richer semantic structure.

Notebook: notebooks/04_cub_gull_tern.ipynb


---

## Setup

### 1. Clone the repository

git clone https://github.com/CRSBanerji/concept-frustration.git

cd concept-frustration


### 2. Create environment

Recommended: Python ≥ 3.9
python -m venv venv
source venv/bin/activate


### 3. Install dependencies

pip install -r requirements.txt


---

## Running the Experiments

Experiments are organised as Jupyter notebooks.

Recommended order:

1. Globe / toy geometry task
2. Synthetic simulations
3. Sarcasm NLP task
4. CUB vision task

Run with:

jupyter notebook

---

## Results

Example result tables are included in the `results/` directory.

These correspond to metrics reported in the manuscript, including:

- black-box model accuracy
- concept bottleneck model accuracy
- frustration metrics
- alignment metrics

---

## Reproducibility Notes

This repository contains research code accompanying the manuscript. Some notebooks are structured for exploratory analysis and may prioritise clarity of experimentation over software modularity.

Key experiment outputs used in the paper are included as CSV files in the `results/` directory.

---

## License

This repository is released under the **MIT License**.

You are free to use, modify, and distribute the code with appropriate attribution.

---

## Citation

If you use this repository or build on this work, please cite:
**Aligning Human Concepts and Machine Representations in Interpretable AI**  
Enrico Parisini, Christopher Soelistyo, Ahab Isaac, Alessandro Barp, Christopher R. S. Banerji
