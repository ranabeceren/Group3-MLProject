# Semester Project: Satellite Data Analysis (Group Project)

This project trains deep learning models to predict **building masks** from **Sentinel‑2 image patches** using **OSM building footprints** as supervision.

## Running the pipeline

All scripts are intended to be executed from the project root:

```bash

The repository is organized into:
- `data_acquisition/` — download Sentinel‑2 + OSM, generate masks
- `deep_learning/` — patch pipeline + training/testing + experiments
- `results/` — saved plots, overlays, and result tables
- `environment-py310.yml` — reproducible environment

---

All scripts are intended to be executed from the project root (`Group3-MLProject/`):

```bash
python3 deep_learning/runner.py
python deep_learning/runner.py

All scripts are intended to be executed from the project root (`Group3-MLProject/`).

The OSM pipeline generates the data in the `data/` directory at the project root.
Due to size constraints, the data are not included in the repository.

## Environment setup

This project supports Python **3.10** and **3.11**.
To use Python 3.10, change `python=3.11` to `python=3.10` in `environment.yml`.

## Setup

Create the environment (Python 3.11 recommended):
```bash
conda env create -f environment-py311.yml
conda activate ml-env-py311

Alternatively, Python 3.10:
conda env create -f environment-py310.yml
conda activate ml-env-py310

# Running commends 
python3 data_acquisition/src/osm/osm_main_pipeline.py 
(this is absolute true)

python3 data_acquisition/src/sentinel/download_fullbands_job.py
(this is absolute true)

python3 data_acquisition/src/sentinel/generate_filters.py
(this is absolute true)

python3 data_acquisition/src/sentinel/mask_generator.py
(this is absolute true)

python3 deep_learning/runner.py
(this is absolute true)

### Loding the data takes about a half an hour.

conda install -y -c conda-forge intel-openmp
conda install -y -c conda-forge intel-openmp tbb
conda install -y -c conda-forge "intel-ittnotify"
