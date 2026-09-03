# First-year statistics project 
CSX 2003-  Principles of Statistics Team Project 

# Statistics Superstars

## Data Detective: Exploring Real-World Distributions

This project explores the **Iris Flower Dataset** using descriptive statistics, probability distributions, confidence intervals, hypothesis testing, ANOVA, correlation analysis, visualisations, and an interactive Streamlit dashboard.

---

## Project Overview

The Iris dataset contains measurements from 150 Iris flowers across three species:

- Setosa
- Versicolor
- Virginica

Each observation contains four numerical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project investigates how these measurements are distributed, how they differ across species, and how strongly the variables are related.

---

## Dataset

**Dataset:** Iris Flower Dataset  
**Source:** UCI Machine Learning Repository  

https://archive.ics.uci.edu/dataset/53/iris

### Dataset Size

- 150 observations
- 4 numerical variables
- 1 categorical variable
- 3 Iris species
- 50 observations per species
- No missing values

---

## Team Members

| Student | Student ID | Primary Role |
|---|---:|---|
| May Thu Khin | 6611281 | Project Lead & Data Curator |
| Mya Thet Htar | 6915066 | Statistical Analyst |
| Nang Htwarr Ne | 6912054 | Visualization Specialist |

All team members contribute to testing, documentation, code review, final presentation, and project completion.

---

## Project Structure

```text
statistics-superstars/
│
├── data/
│   ├── raw/
│   │   └── iris.data
│   └── processed/
│       └── iris_clean.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_visualization_dashboard.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── statistics.py
│   └── visualizations.py
│
├── tests/
│   ├── test_statistics.py
│   └── test_visualizations.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── final_report.md
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── README.md
└── .gitignore

## Interactive Dashboard

### Live Dashboard

[Iris Statistics Dashboard](https://statistics-superstars-lt9bglkv8g3advdudpd8ms.streamlit.app/)
