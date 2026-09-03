# Final Report

## Data Detective: Exploring Real-World Distributions

**Course:** CSX 2002 Principles of Statistics  
**Semester:** 1/2026  
**Team:** Statistics Superstars  

### Team Members

- May Thu Khin (6611281) — Project Lead & Data Curator
- Mya Thet Htar (6915066) — Statistical Analyst
- Nang Htwarr Ne (6912054) — Visualization Specialist

---

## Interactive Dashboard

### Live Dashboard

[Iris Statistics Dashboard](https://statistics-superstars-lt9bglkv8g3advdudpd8ms.streamlit.app/)


----

# 1. Introduction

This project investigates statistical patterns in the Iris Flower Dataset using Python.

The Iris dataset contains measurements of 150 flowers belonging to three species:

- Setosa
- Versicolor
- Virginica

Each flower contains four numerical measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The main purpose of the project is to apply descriptive and inferential statistical methods to real data and examine how flower measurements differ among Iris species.

The project also demonstrates reusable Python functions, automated unit testing, data visualization, and an interactive Streamlit dashboard.

---

# 2. Research Questions

The analysis focuses on the following questions:

1. What are the distributions of the Iris measurements?
2. Are the numerical variables normally distributed?
3. What are the 95% confidence intervals for the mean measurements?
4. Do flower measurements differ significantly between Iris species?
5. Does mean petal length differ across Setosa, Versicolor, and Virginica?
6. What correlations exist between the numerical measurements?
7. Which variables provide the clearest distinction between Iris species?

---

# 3. Dataset

## 3.1 Source

The Iris dataset was obtained from the UCI Machine Learning Repository.

Dataset link:

https://archive.ics.uci.edu/dataset/53/iris

## 3.2 Dataset Structure

The dataset contains:

- 150 observations
- 4 numerical variables
- 1 categorical variable
- 3 Iris species
- 50 observations for each species

The numerical variables are measured in centimeters.

---

# 4. Data Preparation

The raw dataset was loaded and cleaned using the custom Python module:

`src/data_loader.py`

The cleaning process included:

- Assigning appropriate column names
- Removing completely empty rows
- Converting measurement columns to numerical values
- Standardizing species names
- Checking for missing values
- Inspecting duplicate observations
- Saving the cleaned dataset to `data/processed/iris_clean.csv`

No missing values were found.

Three duplicate rows were detected. These observations were retained because identical flower measurements do not necessarily represent data-entry errors. Different flowers may legitimately have identical recorded measurements.

The cleaned dataset therefore contains all 150 original observations.

---

# 5. Descriptive Statistics

Descriptive statistics were calculated for the four numerical variables.

| Variable | Mean | Median | Mode | Std. Dev. | Variance | Range | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sepal Length | 5.843 | 5.80 | 5.0 | 0.828 | 0.686 | 3.6 | 1.3 |
| Sepal Width | 3.054 | 3.00 | 3.0 | 0.434 | 0.188 | 2.4 | 0.5 |
| Petal Length | 3.759 | 4.35 | 1.5 | 1.764 | 3.113 | 5.9 | 3.5 |
| Petal Width | 1.199 | 1.30 | 0.2 | 0.763 | 0.582 | 2.4 | 1.5 |

Petal length shows the largest variability, with a standard deviation of 1.764 cm and a range of 5.9 cm.

This large variation is related to the substantial differences in petal size among the three Iris species.

---

# 6. Distribution and Normality Analysis

Normal distribution fitting was performed using the Kolmogorov-Smirnov goodness-of-fit test.

| Variable | KS P-Value | Interpretation |
|---|---:|---|
| Sepal Length | 0.1706 | No strong evidence against normality |
| Sepal Width | 0.0769 | No strong evidence against normality |
| Petal Length | < 0.0001 | Evidence of non-normality |
| Petal Width | 0.0002 | Evidence of non-normality |

At a significance level of 0.05, sepal length and sepal width do not show strong evidence against normality.

Petal length and petal width show significant departures from a single normal distribution.

This is likely influenced by combining the three Iris species, which have distinctly different petal sizes.

Q-Q plots were also used for visual assessment. Sepal measurements followed the theoretical normal reference line more closely, while petal measurements showed clearer departures.

---

# 7. Bootstrap Confidence Intervals

Bootstrap resampling with 10,000 samples was used to estimate 95% confidence intervals for the population means.

| Variable | Sample Mean | Lower 95% CI | Upper 95% CI |
|---|---:|---:|---:|
| Sepal Length | 5.843 | 5.712 | 5.974 |
| Sepal Width | 3.054 | 2.985 | 3.124 |
| Petal Length | 3.759 | 3.477 | 4.033 |
| Petal Width | 1.199 | 1.077 | 1.317 |

Petal length has the widest confidence interval among the four measurements, which is consistent with its greater variability.

---

# 8. Two-Sample Hypothesis Test

A Welch two-sample t-test was used to compare mean petal length between Setosa and Versicolor.

## Hypotheses

**H₀:** Mean petal length is equal between Setosa and Versicolor.

**H₁:** Mean petal length differs between Setosa and Versicolor.

The significance level was:

α = 0.05

## Results

Mean petal length:

- Setosa = 1.464 cm
- Versicolor = 4.260 cm

Test results:

- t-statistic = -39.4687
- p-value ≈ 1.06 × 10⁻⁴⁵

Since the p-value is much smaller than 0.05, the null hypothesis is rejected.

There is statistically significant evidence that Setosa and Versicolor have different mean petal lengths.

---

# 9. One-Way ANOVA

A one-way ANOVA was performed to test whether mean petal length differs across all three Iris species.

## Hypotheses

**H₀:** Mean petal length is equal across Setosa, Versicolor, and Virginica.

**H₁:** At least one species has a different mean petal length.

Mean petal lengths were:

| Species | Mean Petal Length |
|---|---:|
| Setosa | 1.464 cm |
| Versicolor | 4.260 cm |
| Virginica | 5.552 cm |

ANOVA results:

- F-statistic = 1179.0343
- p-value ≈ 3.05 × 10⁻⁹¹

Since the p-value is much smaller than 0.05, the null hypothesis is rejected.

Therefore, there is statistically significant evidence that mean petal length is not equal across the three Iris species.

---

# 10. ANOVA Assumption Checks

## 10.1 Normality Within Species

Shapiro-Wilk tests were performed separately for each species.

| Species | W Statistic | P-Value |
|---|---:|---:|
| Setosa | 0.9549 | 0.0547 |
| Versicolor | 0.9660 | 0.1585 |
| Virginica | 0.9622 | 0.1098 |

All p-values are greater than 0.05.

Therefore, there is no significant evidence of non-normality within the three petal-length groups.

## 10.2 Equal Variances

Levene's test was performed to assess homogeneity of variance.

Results:

- Test statistic = 19.7201
- p-value < 0.05

The equal-variance assumption is therefore violated.

However, each species contains the same number of observations (50), making ordinary ANOVA relatively robust to this violation.

A variance-robust Welch ANOVA was therefore also performed.

---

# 11. Welch's ANOVA

Welch's ANOVA produced:

- F-statistic = 1826.581
- p-value ≈ 2.85 × 10⁻⁶⁶

The result remains highly statistically significant.

Therefore, the conclusion that mean petal length differs among Iris species remains valid even when unequal variances are taken into account.

---

# 12. Pairwise Comparisons

Pairwise Welch t-tests were conducted following the significant ANOVA result.

A Bonferroni correction was used to account for the three comparisons.

| Comparison | Bonferroni-Adjusted P-Value | Significant |
|---|---:|---|
| Setosa vs Versicolor | 3.17 × 10⁻⁴⁵ | Yes |
| Setosa vs Virginica | 2.91 × 10⁻⁴⁹ | Yes |
| Versicolor vs Virginica | 1.47 × 10⁻²¹ | Yes |

All three comparisons remain statistically significant after adjustment.

Therefore, all three Iris species have significantly different mean petal lengths.

The observed ordering is:

Setosa < Versicolor < Virginica

---

# 13. Correlation Analysis

Correlation analysis showed strong relationships among several flower measurements.

The strongest relationship was between petal length and petal width.

Pearson correlation results:

- r = 0.9628
- p-value ≈ 5.78 × 10⁻⁸⁶

This represents a very strong positive linear relationship.

Therefore, flowers with longer petals generally also have wider petals.

The correlation is statistically significant.

However, correlation represents association and does not establish causation.

---

# 14. Visualizations

The project includes several types of visualizations:

- Histograms with fitted distributions
- Q-Q plots
- Boxplots by Iris species
- Correlation heatmaps
- Scatter plots
- Interactive Plotly scatter plots

The visualizations consistently show that petal measurements provide clearer separation among the three Iris species than sepal measurements.

Setosa has substantially smaller petals, Versicolor generally has intermediate petal measurements, and Virginica has the largest petals.

---

# 15. Interactive Dashboard

An interactive Streamlit dashboard was developed to allow users to explore the analysis dynamically.

The dashboard allows users to:

- Filter Iris species
- Select measurements
- View descriptive statistics
- Examine fitted distributions
- View Q-Q plots
- Compare species through boxplots
- Explore interactive scatter plots
- Examine correlations
- Calculate bootstrap confidence intervals
- Run Welch two-sample t-tests
- Run one-way ANOVA
- View and download filtered data

The dashboard uses reusable functions from the project's statistical and visualization modules.

---

# 16. Code Quality and Testing

The project follows a modular Python structure.

The main modules are:

- `src/data_loader.py`
- `src/statistics.py`
- `src/visualizations.py`
- `dashboard/app.py`

Automated unit tests are stored in:

- `tests/test_statistics.py`
- `tests/test_visualizations.py`

The test suite checks statistical calculations, visualization functions, dashboard configuration, and error handling.

At the final testing stage:

**12 automated tests passed successfully.**

---

# 17. Key Findings

The major findings from the project are:

1. Petal measurements show substantially greater differences across species than sepal measurements.
2. Setosa has the smallest petals, Versicolor has intermediate petal measurements, and Virginica has the largest.
3. Mean petal length differs significantly among all three Iris species.
4. The ANOVA conclusion remains significant when unequal variances are addressed using Welch's ANOVA.
5. Every pair of species differs significantly in mean petal length after Bonferroni correction.
6. Petal length and petal width have an extremely strong positive correlation.
7. Petal variables provide particularly useful information for distinguishing Iris species.

---

# 18. Limitations

Several limitations should be considered.

First, the Iris dataset contains only 150 observations and only three Iris species.

Second, the dataset includes only four flower measurements. Other biological or environmental variables are not available.

Third, the dataset is a classic educational dataset and may not represent the full natural variation of Iris populations.

Fourth, correlation should not be interpreted as evidence that one flower measurement causes another.

Finally, statistical significance does not automatically imply practical importance. Results should therefore be interpreted together with effect size, visual patterns, and subject-matter context.

---

# 19. Conclusion

The Iris dataset provides clear statistical evidence of differences among Setosa, Versicolor, and Virginica.

Petal length and petal width are particularly informative because their measurements differ strongly across species and are highly correlated.

The one-way ANOVA found a highly significant difference in mean petal length among the three species. Although the equal-variance assumption was violated, Welch's ANOVA confirmed the same conclusion.

Pairwise testing further showed that all three species differ significantly from one another in mean petal length.

Overall, the project demonstrates how descriptive statistics, probability distributions, confidence intervals, hypothesis testing, ANOVA, correlation analysis, visualization, and interactive tools can be combined to investigate patterns in real-world data.

---

# 20. Team Contributions

## May Thu Khin — Project Lead & Data Curator

Primary responsibilities:

- Repository setup and maintenance
- Dataset acquisition and preparation
- Data cleaning workflow
- Project coordination
- Final report compilation
- Integration of project components

## Mya Thet Htar — Statistical Analyst

Primary responsibilities:

- Statistical methods
- Descriptive statistics
- Confidence intervals
- Hypothesis testing
- ANOVA analysis
- Statistical interpretation

## Nang Htwarr Ne — Visualization Specialist

Primary responsibilities:

- Data visualization
- Plotting functions
- Interactive visualizations
- Dashboard visualization design
- Visual accessibility and presentation

All team members contribute to testing, documentation, code review, project review, and the final presentation.

---

# 21. Repository

GitHub Repository:

https://github.com/Mtkhin/statistics-superstars

---

# 22. References

UCI Machine Learning Repository. Iris Dataset.

https://archive.ics.uci.edu/dataset/53/iris

SciPy Documentation.

https://docs.scipy.org/doc/scipy/reference/stats.html

Pandas Documentation.

https://pandas.pydata.org/

Streamlit Documentation.

https://docs.streamlit.io/

Plotly Python Documentation.

https://plotly.com/python/