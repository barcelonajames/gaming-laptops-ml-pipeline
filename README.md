# Gaming Laptops Q1 2026 — End-to-End ML Pipeline
### Regression · Classification · Clustering · NLP · Zero-Shot AI

A complete data science pipeline built on 614 Amazon gaming laptop and PC listings scraped in Q1 2026. The project runs the full stack: exploratory analysis, data cleaning, feature engineering, NLP spec extraction, zero-shot classification via Claude Haiku, supervised ML (regression + classification), unsupervised clustering (KMeans), and dimensionality reduction (PCA).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m8SSyW4bHE24XM7iE7CmiHegTZj1Sfvb)
---

## The Finding

> *"In Q1 2026, the average buyer of a gaming-category laptop is not actually a gamer — they are someone looking for a well-reviewed mainstream machine around $1,150, where Apple dominates customer satisfaction and brand trust outranks raw GPU power."*

Four of the five most-reviewed, highest-rated products in a dataset labeled "gaming laptops" are Apple MacBooks with no gaming GPU. The pipeline didn't just model the data — it revealed that the market category itself is misleading.

---

## Dataset

**Source:** Scraped from Amazon (Q1 2026) — included in this repository

| Detail | Value |
|---|---|
| Rows | ~614 product listings |
| Columns | Price, brand, ratings, review count, product title, specs |
| File | `gaming_laptops_2026_q1.csv` |

The dataset is included directly in the repo — it is original scraped data, not a third-party source.

---

## Pipeline Overview

| Step | Technique | Output |
|---|---|---|
| 2 | EDA + Visualization | Distributions, brand breakdown, correlations |
| 3 | Data Cleaning | Zero missing values in key columns; standardized brand names |
| 4a | Feature Engineering | `product_type`, `price_tier`, `has_discount`, `has_description` |
| 4b | NLP: Regex Extraction | `gpu_brand`, `gpu_tier`, `ram_gb`, `refresh_hz`, `cpu_brand`, `display_type` |
| 5 | Zero-Shot (Claude Haiku) | `ai_target_audience` label for sampled products |
| 6 | Label Encoding | All categorical columns converted to integers |
| 7 | Split & Scale | 80/20 split; StandardScaler on training set only |
| 8a | Linear Regression | Predicted price — R² and MAE |
| 8b | Random Forest Classifier | Predicted price tier — precision, recall, F1, confusion matrix |
| 8c | KMeans + Elbow Method | 4 natural market segments identified |
| 8d | PCA | 2D cluster visualization |
| 9 | Export | `gaming_laptops_analyzed.csv` |

---

## Key Findings

**Verdict 1 — Specs alone don't explain price (R² = 0.43)**
Linear regression with an MAE of ~$431 shows that specifications explain less than half of price variation. Brand prestige, retailer markups, and marketing account for the rest. Laptop pricing is not purely rational.

**Verdict 2 — Buyers gravitate $300 below the dataset average**
Products with 50+ reviews (the closest signal to actual purchases) have a median price of $1,149. The dataset's overall median is $1,449. The real market sits firmly in mid-range and budget tiers.

**Verdict 3 — Top 5 features driving price tier**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Review count | 17.3% |
| 2 | RAM | 14.3% |
| 3 | Brand | 13.1% |
| 4 | Refresh rate | 12.3% |
| 5 | Star rating | 11.0% |

GPU brand ranks 6th at 8.4% — surprisingly low for a "gaming" laptop dataset.

**Verdict 4 — Four real market segments from clustering**

| Cluster | Size | Avg Price | Profile |
|---|---|---|---|
| 3 | 413 products (67%) | $1,311 | Mainstream — 16GB RAM, 144-165Hz, 4.5 stars |
| 0 | 73 products | $2,551 | Workstation — ultra-high RAM, creators not gamers |
| 2 | 104 products | $2,448 | True premium gaming — 32GB+, 240Hz, OLED |
| 1 | 24 products | $1,325 | Viral hits — 1,095 avg reviews, high-volume mainstream |

**Verdict 5 — The actual winners are MacBooks**
The top reviewed, highest-rated products in a gaming laptop dataset are Apple MacBooks ($949–$1,599) with no gaming GPU. Only one true gaming laptop made the top 5: the Acer Predator with RTX 5070 Ti, 32GB, OLED 240Hz at $2,100.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/barcelonajames/gaming-laptops-ml-pipeline.git
cd gaming-laptops-ml-pipeline
```

### 2. Set up the environment
```bash
conda create -n gaming-laptops python=3.11 -y
conda activate gaming-laptops
pip install -r requirements.txt
```

### 3. Add your API key
The notebook uses Claude Haiku for zero-shot classification (Step 5). Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```
Get a key at [console.anthropic.com](https://console.anthropic.com). Step 5 can be skipped if no key is available — the rest of the pipeline runs independently.

### 4. Open the notebook
```bash
jupyter notebook gaming_laptops_pipeline.ipynb
```

---

## Project Structure

```
gaming-laptops-ml-pipeline/
├── gaming_laptops_pipeline.ipynb   ← Main notebook (run this)
├── gaming_laptops_2026_q1.csv      ← Raw dataset (included)
├── requirements.txt                ← Python dependencies
├── .gitignore                      ← .env excluded
├── LICENSE                         ← MIT
└── README.md
```

---

## Tools & Libraries

| Category | Libraries |
|---|---|
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn |
| NLP | re (regex), Hugging Face transformers |
| Zero-shot AI | Anthropic Claude Haiku (via API) |
| Environment | Python 3.11, conda, python-dotenv |

---

## Context

Built as part of the **Uplift Code Camp Python for Data and AI Bootcamp** (2026). This project is distinct from the others in the portfolio because it incorporates a live LLM API call (Claude Haiku) as part of the pipeline — demonstrating how AI classification can augment traditional ML without requiring labeled training data.

---

*James Aleister Barcelona — Visual Designer & Data Analyst | Davao, Philippines*
