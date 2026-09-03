import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------------------------
# Project setup
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.statistics import (
    descriptive_stats,
    fit_distribution,
    bootstrap_ci,
    hypothesis_test,
    anova_test
)

from src.visualizations import (
    plot_histogram_with_distribution,
    create_correlation_heatmap,
    plot_boxplots_by_category,
    create_interactive_scatter,
    plot_qq_comparison,
    dashboard_layout
)


# -------------------------------------------------
# Page configuration
# -------------------------------------------------

st.set_page_config(
    **dashboard_layout()
)


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

@st.cache_data
def load_data():

    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "iris_clean.csv"
    )

    return pd.read_csv(
        data_path
    )


df = load_data()

numeric_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]


# -------------------------------------------------
# Header
# -------------------------------------------------

st.title(
    "🌸 Iris Statistics Dashboard"
)

st.write(
    """
    Interactive statistical exploration of the Iris flower
    dataset. Explore distributions, species differences,
    relationships, confidence intervals, and hypothesis tests.
    """
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header(
    "Dashboard Controls"
)

selected_species = st.sidebar.multiselect(
    "Select Iris species",
    options=sorted(
        df["species"].unique()
    ),
    default=sorted(
        df["species"].unique()
    )
)

selected_variable = st.sidebar.selectbox(
    "Select measurement",
    numeric_columns,
    index=2
)

scatter_x = st.sidebar.selectbox(
    "Scatter plot X-axis",
    numeric_columns,
    index=2
)

scatter_y = st.sidebar.selectbox(
    "Scatter plot Y-axis",
    numeric_columns,
    index=3
)


# -------------------------------------------------
# Filter data
# -------------------------------------------------

if selected_species:

    filtered_df = df[
        df["species"].isin(
            selected_species
        )
    ].copy()

else:

    filtered_df = df.copy()


# -------------------------------------------------
# Overview
# -------------------------------------------------

st.subheader(
    "Dataset Overview"
)

overview1, overview2, overview3, overview4 = (
    st.columns(4)
)

overview1.metric(
    "Observations",
    len(filtered_df)
)

overview2.metric(
    "Selected Species",
    filtered_df["species"].nunique()
)

overview3.metric(
    "Mean",
    f"{filtered_df[selected_variable].mean():.3f} cm"
)

overview4.metric(
    "Standard Deviation",
    f"{filtered_df[selected_variable].std():.3f} cm"
)


with st.expander(
    "View Dataset"
):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# -------------------------------------------------
# Descriptive statistics
# -------------------------------------------------

st.subheader(
    "Descriptive Statistics"
)

stats_result = descriptive_stats(
    filtered_df[
        selected_variable
    ].tolist()
)

stats_table = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Median",
        "Mode",
        "Standard Deviation",
        "Variance",
        "Range",
        "IQR",
        "Skewness",
        "Kurtosis"
    ],

    "Value": [
        stats_result["mean"],
        stats_result["median"],
        stats_result["mode"],
        stats_result["std"],
        stats_result["variance"],
        stats_result["range"],
        stats_result["iqr"],
        stats_result["skewness"],
        stats_result["kurtosis"]
    ]
})

stats_table["Value"] = (
    stats_table["Value"]
    .round(3)
)

st.dataframe(
    stats_table,
    use_container_width=True,
    hide_index=True
)


# -------------------------------------------------
# Distribution analysis
# -------------------------------------------------

st.subheader(
    "Distribution Analysis"
)

distribution_left, distribution_right = (
    st.columns(2)
)

with distribution_left:

    st.markdown(
        "#### Histogram + Normal Distribution"
    )

    histogram_fig = (
        plot_histogram_with_distribution(
            filtered_df,
            selected_variable,
            distribution="normal"
        )
    )

    st.pyplot(
        histogram_fig
    )


with distribution_right:

    st.markdown(
        "#### Q-Q Plot"
    )

    qq_fig = plot_qq_comparison(
        filtered_df,
        selected_variable
    )

    st.pyplot(
        qq_fig
    )


# -------------------------------------------------
# Normal distribution fitting
# -------------------------------------------------

st.markdown(
    "#### Normal Distribution Fit"
)

normal_result = fit_distribution(
    filtered_df[
        selected_variable
    ].tolist(),
    "normal"
)

normal_mean = (
    normal_result["params"][0]
)

normal_std = (
    normal_result["params"][1]
)

normal_ks = (
    normal_result["ks_statistic"]
)

normal_p = (
    normal_result["p_value"]
)

normal1, normal2, normal3, normal4 = (
    st.columns(4)
)

normal1.metric(
    "Fitted Mean",
    f"{normal_mean:.3f}"
)

normal2.metric(
    "Fitted Std Dev",
    f"{normal_std:.3f}"
)

normal3.metric(
    "KS Statistic",
    f"{normal_ks:.4f}"
)

normal4.metric(
    "P-Value",
    f"{normal_p:.3e}"
)


if normal_p > 0.05:

    st.info(
        "There is not enough evidence to reject "
        "normality at α = 0.05."
    )

else:

    st.warning(
        "The selected data shows evidence of "
        "departure from normality at α = 0.05."
    )


# -------------------------------------------------
# Boxplots
# -------------------------------------------------

st.subheader(
    "Comparison Across Species"
)

boxplot_fig = (
    plot_boxplots_by_category(
        filtered_df,
        selected_variable,
        "species"
    )
)

st.pyplot(
    boxplot_fig
)


# -------------------------------------------------
# Interactive scatter plot
# -------------------------------------------------

st.subheader(
    "Interactive Relationship Between Measurements"
)

scatter_fig = (
    create_interactive_scatter(
        filtered_df,
        scatter_x,
        scatter_y,
        "species"
    )
)

st.plotly_chart(
    scatter_fig,
    use_container_width=True
)


# -------------------------------------------------
# Correlation heatmap
# -------------------------------------------------

st.subheader(
    "Correlation Analysis"
)

correlation_fig = (
    create_correlation_heatmap(
        filtered_df,
        numeric_columns
    )
)

st.pyplot(
    correlation_fig
)


with st.expander(
    "View Correlation Matrix"
):

    correlation_matrix = (
        filtered_df[
            numeric_columns
        ]
        .corr()
        .round(3)
    )

    st.dataframe(
        correlation_matrix,
        use_container_width=True
    )


# -------------------------------------------------
# Species averages
# -------------------------------------------------

st.subheader(
    "Mean Measurements by Species"
)

species_summary = (
    filtered_df
    .groupby(
        "species"
    )[numeric_columns]
    .mean()
    .round(3)
)

st.dataframe(
    species_summary,
    use_container_width=True
)


# -------------------------------------------------
# Confidence intervals
# -------------------------------------------------

st.subheader(
    "95% Bootstrap Confidence Interval"
)

ci_lower, ci_upper = (
    bootstrap_ci(
        filtered_df[
            selected_variable
        ].tolist(),
        np.mean,
        n_bootstrap=5000,
        confidence=0.95
    )
)

ci1, ci2, ci3 = (
    st.columns(3)
)

ci1.metric(
    "Sample Mean",
    f"{filtered_df[selected_variable].mean():.3f} cm"
)

ci2.metric(
    "Lower 95% CI",
    f"{ci_lower:.3f} cm"
)

ci3.metric(
    "Upper 95% CI",
    f"{ci_upper:.3f} cm"
)


# -------------------------------------------------
# Hypothesis testing
# -------------------------------------------------

st.subheader(
    "Hypothesis Testing"
)

st.write(
    """
    Welch's two-sample t-test compares the selected
    measurement between two Iris species.
    """
)

species_options = sorted(
    df["species"].unique()
)

test1, test2 = (
    st.columns(2)
)

with test1:

    group1_name = st.selectbox(
        "First species",
        species_options,
        index=0,
        key="test_group_1"
    )


with test2:

    group2_name = st.selectbox(
        "Second species",
        species_options,
        index=1,
        key="test_group_2"
    )


if group1_name == group2_name:

    st.warning(
        "Please choose two different species."
    )

else:

    group1_data = df[
        df["species"]
        == group1_name
    ][selected_variable].tolist()

    group2_data = df[
        df["species"]
        == group2_name
    ][selected_variable].tolist()

    ttest_result = (
        hypothesis_test(
            group1_data,
            group2_data,
            test_type="t-test"
        )
    )

    test_result1, test_result2, test_result3 = (
        st.columns(3)
    )

    test_result1.metric(
        f"{group1_name.title()} Mean",
        f"{np.mean(group1_data):.3f} cm"
    )

    test_result2.metric(
        f"{group2_name.title()} Mean",
        f"{np.mean(group2_data):.3f} cm"
    )

    test_result3.metric(
        "P-Value",
        f"{ttest_result['p_value']:.3e}"
    )

    st.write(
        f"**T-statistic:** "
        f"{ttest_result['statistic']:.4f}"
    )

    if (
        ttest_result[
            "p_value"
        ] < 0.05
    ):

        st.success(
            "Reject H₀: A statistically significant "
            "difference exists between the two species."
        )

    else:

        st.info(
            "Fail to reject H₀: No statistically "
            "significant difference was detected."
        )


# -------------------------------------------------
# One-way ANOVA
# -------------------------------------------------

st.subheader(
    "One-Way ANOVA"
)

anova_groups = []

for species in species_options:

    species_values = df[
        df["species"]
        == species
    ][selected_variable].tolist()

    anova_groups.append(
        species_values
    )


anova_result = (
    anova_test(
        *anova_groups
    )
)

anova1, anova2 = (
    st.columns(2)
)

anova1.metric(
    "F-Statistic",
    f"{anova_result['f_statistic']:.3f}"
)

anova2.metric(
    "P-Value",
    f"{anova_result['p_value']:.3e}"
)


if (
    anova_result[
        "p_value"
    ] < 0.05
):

    st.success(
        "Reject H₀: At least one Iris species "
        "has a significantly different mean."
    )

else:

    st.info(
        "Fail to reject H₀: No statistically significant "
        "difference was detected across species."
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Statistics Superstars – "
    "Data Detective: Exploring Real-World Distributions"
)