# Detecting Potentially Misleading Climate Claims in Danish Media

This repository contains the code for a master's thesis estimating the prevalence of potentially misleading climate claims in Danish journalism, and assessing whether a lightweight classifier can support their identification.

## Overview

The project works with a corpus of 182,584 Danish climate-related news articles (Infomedia, 2023–2025) and pursues two objectives:

1. **Prevalence estimation.** Weak labels are generated for an 18,000-article sample using the CARDS taxonomy, then corrected for systematic annotation bias using Design-Based Supervised Learning (DSL). This yields a bias-corrected prevalence estimate of approximately **9.9%**, compared with a naive estimate of 95%.

2. **Classifier feasibility.** A lightweight Danish BERT classifier is trained through active learning to assess whether such claims can be identified at scale under resource constraints.

## Repository structure

```
src/
├── preprocessing/                    # Corpus cleaning and sample construction
├── annotation/                       # CARDS weak labelling and manual annotation
├── data/                             # Datasets construction
├── design-based supervised learning/ # DSL bias correction and prevalence estimation
├── training/                         # Active learning and classifier training
└── evaluation/                       # Performance metrics and analysis
environment.yml                       # Conda environment specification
```

## Setup

```bash
conda env create -f environment.yml
conda activate <climate-misinfo>
```

## Data availability

The article corpus is drawn from the Infomedia archive under restricted access and is therefore not included in this repository.

## Author

Frederikke Buchsti Hermansen
