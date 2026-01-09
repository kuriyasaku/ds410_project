# WineEnthusiast ML Pipeline — README

> **Status**: Feature pipeline & ML-ready dataset complete (week of update). Baseline modeling begins next week.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Preprocessing & Standardization](#preprocessing--standardization)
* [Feature Engineering](#feature-engineering)
* [Engineering & Scale](#engineering--scale)
* [Modeling Plan](#modeling-plan)
* [Visualization Plan](#visualization-plan)
* [Evaluation & Scalability Plan](#evaluation--scalability-plan)
* [Project Structure](#project-structure)
* [Quickstart](#quickstart)
* [Reproducibility Notes](#reproducibility-notes)
* [Outputs](#outputs)
* [Acknowledgements](#acknowledgements)

---

## Overview

End-to-end PySpark pipeline for cleaning, standardizing, and featurizing the **Kaggle WineEnthusiast** ratings dataset, producing an **ML-ready Parquet** for baseline modeling (regression, classification, and clustering). Pipeline emphasizes **scalability**, **compression**, and **shareability**.

---

## Dataset

* **Source**: Kaggle WineEnthusiast ratings dataset.
* **Raw Size**: ~130k rows.
* **Working Size after cleaning**: ~120k rows.

> The project treats the Kaggle export as the only raw input and builds all artifacts downstream from it.

---

## Preprocessing & Standardization

**Goals**: rectify header quirks, address missingness, normalize string fields, and deliver a clean, partitioned Parquet.

* **Header & Missing Data Handling**

  * Correct/ignore blank first column in CSV.
  * Rule-based imputation or filtering for missing: `price`, `points`, `designation`, `region_x`.
  * Text cleanup for `description`.
* **Normalization**

  * Canonicalized fields: `country_norm`, `province_norm`, `region_1_norm`, `region_2_norm`, `variety_norm`, `winery_norm`, `taster_name_norm`.
  * Outlier control: `price_winsor` (winsorized price).
* **Artifacts**

  * Clean, partitioned Parquet at `outputs/wine_clean.parquet` (multi-part; ~120k rows).
  * Optional merged CSV exports at `outputs/wine_clean_csv/` for Excel preview.
* **Implementation**

  * All steps tracked in PySpark script/notebook: `wine_preprocessing_pyspark_c`.

---

## Feature Engineering

* **Categorical**: `StringIndexer` + `OneHotEncoder` (e.g., `country_norm`, `province_norm`, `variety_canonical`).
* **Numerical**: `points`, `price_winsor` (with optional normalization/standardization flags for later experiments).
* **Text**: `description` → TF–IDF → **PCA to 500 dims** (compressing 8k+ term features to 500 to reduce training/storage cost).
* **Assembly**: `VectorAssembler` to a unified `features` vector.
* **Artifact**: ML-ready dataset at `outputs/wine_ml_ready.parquet`.

---

## Engineering & Scale

* **Spark runtime tuning**

  * Configure driver/executor memory, `memoryFraction`, shuffle partitions.
  * Persist with `StorageLevel.MEMORY_AND_DISK` and use sensible `repartition` for downstream performance.
* **I/O & Compression**

  * Parquet with **Snappy** compression; **64MB** Parquet block size.

---

## Modeling Plan

Baselines reuse the unified `features` vector for comparability.

* **Regression**: predict `points` or `price`

  * `LinearRegression`, `GBTRegressor`, `RandomForestRegressor`
* **Binary Classification**: high vs. low score

  * `LogisticRegression`, `GBTClassifier`
* **Unsupervised Clustering**

  * `KMeans` for flavor/region similarity grouping

**Model directory** prepared at `outputs/models/`.

---

## Visualization Plan

* Price–score relationship
* Country/variety distributions
* Feature importance (tree-based models)
* Clustering projections + example interpretations

---

## Evaluation & Scalability Plan

* Compare **training/inference latency**, **memory**, and **metrics** across **1/2/4 cores**.
* Assess cost-effectiveness across sampling scales.

---

## Project Structure

```
.
├─ data/
│  └─ raw/                         # Raw Kaggle export(s)
├─ notebooks/
│  └─ wine_preprocessing_pyspark_c.ipynb   # or .py script equivalent
├─ outputs/
│  ├─ wine_clean.parquet/          # partitioned Parquet (clean)
│  ├─ wine_clean_csv/              # optional merged CSVs for sharing
│  ├─ wine_ml_ready.parquet/       # ML-ready features
│  └─ models/                      # trained models, metrics, artifacts
├─ src/
│  └─ preprocessing/               # reusable transforms (optional)
├─ conf/
│  └─ spark.conf                   # example Spark config (optional)
└─ README.md
```

---

## Quickstart

### 1) Environment

* **Python**: 3.9+
* **Spark**: 3.3+ (compatible with your cluster setup)

### 2) Suggested Spark settings (local example)

```bash
spark-submit \
  --master local[4] \
  --driver-memory 8g \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.memory.fraction=0.6 \
  src/preprocess.py \
  --input data/raw/winemag.csv \
  --clean outputs/wine_clean.parquet \
  --ml outputs/wine_ml_ready.parquet
```

### 3) Notebook usage

Open `notebooks/wine_preprocessing_pyspark_c.ipynb` and run all cells. Artifacts will be written to `outputs/`.

---

## Reproducibility Notes

* All featurization parameters (indexers, encoders, TF–IDF, PCA dims, assembler schemas) should be versioned in code or logged alongside artifacts.
* Keep Kaggle raw export checksum/hashes when possible.
* Pin Spark and PySpark versions for stable serialization and model compatibility.

---

## Outputs

* `outputs/wine_clean.parquet/` — clean, partitioned dataset (~120k rows)
* `outputs/wine_clean_csv/` — optional merged CSV exports for sharing
* `outputs/wine_ml_ready.parquet/` — unified `features` column (categorical + numerical + text-PCA)
* `outputs/models/` — model checkpoints & metrics (to be populated)

---

## Acknowledgements

* Kaggle WineEnthusiast dataset and community contributors.
* Apache Spark & PySpark.
