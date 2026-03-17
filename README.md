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

---

## Repository Structure

The repository is organised as follows:

- `notebooks/`  
  Jupyter notebooks implementing the main experiments (lightweight runners).

- `src/`  
  Modularised code for each experiment, including:
  - data generation / loading  
  - model definitions  
  - training routines  
  - evaluation metrics  
  - experiment pipelines  

- `data/`  
  External datasets (download separately, see below).

- `results/`  
  Output CSV files containing experiment results and metrics used in the manuscript.
---

## Data Setup

This repository does **not** include the large datasets. Please download them manually as follows.

### 1. Sarcasm Headlines Dataset (NLP)

Download from Kaggle:  
https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

Reference:  
Misra, R. (2018). *News Headlines Dataset for Sarcasm Detection*.

After downloading, place the files here:


`data/headlines_data/`
 
- `Sarcasm_Headlines_Dataset.json`
- `Sarcasm_Headlines_Dataset_v2.json`


---

### 2. CUB-200-2011 Dataset (Vision)

Download from Caltech:  
https://data.caltech.edu/records/65de6-vp158

Reference:  
Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011).  
*The Caltech-UCSD Birds-200-2011 Dataset.*

After downloading, extract and place the directory:


`data/CUB_200_2011/`


So your structure should be:


`data/`
- `CUB_200_2011/`
- `headlines_data/`


---

## Experiments

### 1. Globe / Treasure Hunter Toy Environment

A geometric environment illustrating concept misalignment and frustration.

Notebook:  
`notebooks/01_globe_treasure_hunter.ipynb`

---

### 2. Synthetic Concept Geometry Simulations

Controlled simulations studying alignment between concepts, representations, and task structure.

Notebook:  
`notebooks/02_synthetic_simulations.ipynb`

---

### 3. Sarcasm Detection (NLP Task)

Uses language model embeddings to study concept frustration in natural language.

Notebook:  
`notebooks/03_sarcasm_task.ipynb`

---

### 4. CUB Bird Classification (Vision Task)

Uses vision model embeddings to study concept frustration in a vision setting.

Notebook:  
`notebooks/04_cub_gull_tern.ipynb`

---

## Setup

### 1. Clone the repository


`git clone https://github.com/CRSBanerji/concept-frustration.git`

`cd concept-frustration`


### 2. Create environment

Recommended: Python ≥ 3.9

### 3. Install dependencies


`pip install -r requirements.txt`


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

Outputs include:

- black-box model accuracy  
- concept bottleneck model accuracy  
- frustration metrics  

### Frustration Metrics

The repository reports the primary frustration metrics used in the paper as:

- `F_pair_raw_mean` → corresponds to **γ_F** (Fisher-based frustration)  
- `E_pair_raw_mean` → corresponds to **γ_E** (Euclidean-based frustration)  

These are computed as pairwise frustration measures between learned concept representations.

In addition, the code outputs **alternative and auxiliary variants** of these metrics, including:

- normalised versions  
- trimmed / thresholded variants  
- geometry-specific comparisons  

These additional metrics are provided to support robustness analyses and ablations.

### Output Locations

Each experiment writes results to:

`results/<task_name>/`


For example:

- `results/Globe/`
- `results/Synthetic/`
- `results/Sarcasm/`
- `results/CUB/`
---

## Reproducibility Notes

- Datasets are loaded directly from `data/` (no caching is used)
- Results are generated deterministically given seeds
- Notebooks act as experiment entry points; core logic lives in `src/`

This design ensures clarity and reproducibility for research use.

---

## License

This repository is released under the **MIT License**.

You are free to use, modify, and distribute the code with appropriate attribution.

---

## Citation

If you use this repository or build on this work, please cite:

**Aligning Human Concepts and Machine Representations in Interpretable AI**  
Enrico Parisini, Christopher Soelistyo, Ahab Isaac, Alessandro Barp, Christopher R. S. Banerji

---
