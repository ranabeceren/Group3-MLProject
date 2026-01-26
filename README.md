# Semester Project: Satellite Data Analysis (Group Project)

This project trains deep learning models to predict **building masks** from **Sentinel‑2 image patches** using **OSM building footprints** as supervision.

```bash

The repository is organized into:
- `data_acquisition/` — download Sentinel‑2 + OSM, generate masks
- `deep_learning/` — patch pipeline + training/testing + experiments
- `results/` — saved plots, overlays, and result tables
- `environment-py310.yml` — reproducible environment

---

All scripts are intended to be executed from the project root (`Group3-MLProject/`).

The OSM pipeline generates the data in the `data/` directory at the project root.
Due to size constraints, the data are not included in the repository.

## Environment setup

This project supports Python **3.10**.

## Setup

Create the environment:
```bash
conda env create -f environment-py310.yml
conda activate ml-env-py310

# Running commends 
python3 data_acquisition/src/osm/osm_main_pipeline.py 

python3 data_acquisition/src/sentinel/download_fullbands_job.py

python3 data_acquisition/src/sentinel/generate_filters.py

python3 data_acquisition/src/sentinel/mask_generator.py

python3 deep_learning/runner.py
