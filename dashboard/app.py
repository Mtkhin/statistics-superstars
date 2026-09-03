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
# Page setup
# -------------------------------------------------

st.set_page_config(
    **dashboard_layout()
)


# -------------------------------------------------
# Data
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
# Helpers
# -------------------------------------------------

def pretty_label(value: str) -> str:
    return value.replace("_", " ").title()


def format_p_value(value: float) -> str:

    if value < 0.001:
        return f"{value:.2e}"

    return f"{value:.4f}"


# -------------------------------------------------
# Styling
# -------------------------------------------------

st.markdown(
    """
    <style>

    /* PAGE */

    .block-container {
    max-width: 1280px;
    padding-top: 4.2rem;
    padding-bottom: 4rem;
}

    /* REMOVE EXCESS VISUAL NOISE */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* TYPOGRAPHY */

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    h2 {
        margin-top: 0.4rem;
    }

    /* EDITORIAL HEADER */

    .project-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #C4A66A;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 1rem;
    line-height: 1.4;
}

    .project-title {
        color: #EEEAE2;
        font-size: 3.15rem;
        font-weight: 600;
        letter-spacing: -0.045em;
        line-height: 1.04;
        margin-bottom: 0.8rem;
    }

    .project-subtitle {
        max-width: 760px;
        color: #A4AAA3;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1.7rem;
    }

    .header-rule {
        border-top: 1px solid #30352F;
        margin-bottom: 1.5rem;
    }

    /* SECTION COPY */

    .section-label {
        color: #C4A66A;
        font-size: 0.72rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .section-copy {
        color: #A4AAA3;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }

    /* INSIGHT */

    .insight {
        border-left: 2px solid #C4A66A;
        padding: 0.25rem 0 0.25rem 1rem;
        margin: 1.3rem 0 1.5rem 0;
    }

    .insight-label {
        color: #8B918A;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }

    .insight-text {
        color: #EEEAE2;
        font-size: 1rem;
        line-height: 1.55;
    }

    /* METRICS */

    div[data-testid="stMetric"] {
        background: #171B18;
        border: 1px solid #30352F;
        border-radius: 4px;
        padding: 1rem 1rem 0.9rem 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #929991;
    }

    div[data-testid="stMetricValue"] {
        color: #EEEAE2;
        font-weight: 500;
    }

    /* TABS */

    div[data-baseweb="tab-list"] {
        gap: 1.6rem;
        border-bottom: 1px solid #30352F;
    }

    button[data-baseweb="tab"] {
        background: transparent;
        padding-left: 0;
        padding-right: 0;
        color: #929991;
        font-weight: 500;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #EEEAE2;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #C4A66A;
    }

    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #30352F;
    }

    /* TABLES */

    div[data-testid="stDataFrame"] {
        border: 1px solid #30352F;
    }

    /* DIVIDERS */

    hr {
        border-color: #30352F;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Header
# -------------------------------------------------

st.markdown(
    """
    <div class="project-kicker">
        Statistics Superstars / Iris Study
    </div>

    <div class="project-title">
        Patterns in the Iris Dataset
    </div>

    <div class="project-subtitle">
        An exploratory statistical study of how sepal and petal
        measurements vary across Setosa, Versicolor, and Virginica.
        The analysis combines descriptive statistics, distributions,
        confidence intervals, correlation, and hypothesis testing.
    </div>

    <div class="header-rule"></div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.markdown(
    "## Explore"
)

selected_species = st.sidebar.multiselect(
    "Species",
    options=sorted(
        df["species"].unique()
    ),
    default=sorted(
        df["species"].unique()
    ),
    format_func=lambda value: value.title()
)

selected_variable = st.sidebar.selectbox(
    "Measurement",
    options=numeric_columns,
    index=2,
    format_func=pretty_label
)

scatter_x = st.sidebar.selectbox(
    "Horizontal axis",
    options=numeric_columns,
    index=2,
    format_func=pretty_label
)

scatter_y = st.sidebar.selectbox(
    "Vertical axis",
    options=numeric_columns,
    index=3,
    format_func=pretty_label
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "CSX 2003 Principles of Statistics - Term Project"
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


analysis_species = sorted(
    filtered_df["species"].unique()
)


# -------------------------------------------------
# Dynamic insight
# -------------------------------------------------

corr_matrix = (
    filtered_df[
        numeric_columns
    ]
    .corr()
)

lower_triangle = corr_matrix.where(
    np.tril(
        np.ones(
            corr_matrix.shape
        ),
        k=-1
    ).astype(bool)
)

strongest_pair = (
    lower_triangle
    .abs()
    .stack()
    .idxmax()
)

strongest_r = corr_matrix.loc[
    strongest_pair[0],
    strongest_pair[1]
]


# -------------------------------------------------
# Tabs
# -------------------------------------------------

tab_overview, tab_distribution, tab_relationships, tab_inference, tab_data = (
    st.tabs(
        [
            "Overview",
            "Distributions",
            "Relationships",
            "Tests & inference",
            "Dataset"
        ]
    )
)


# =================================================
# OVERVIEW
# =================================================

with tab_overview:

    st.markdown(
        '<div class="section-label">At a glance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-copy">'
        'A concise view of the dataset currently selected.'
        '</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Observations",
        len(filtered_df)
    )

    metric2.metric(
        "Species",
        filtered_df[
            "species"
        ].nunique()
    )

    metric3.metric(
        f"Mean {pretty_label(selected_variable)}",
        f"{filtered_df[selected_variable].mean():.3f} cm"
    )

    metric4.metric(
        "Standard deviation",
        f"{filtered_df[selected_variable].std():.3f} cm"
    )

    st.markdown(
        f"""
        <div class="insight">
            <div class="insight-label">
                Key relationship
            </div>
            <div class="insight-text">
                {pretty_label(strongest_pair[0])} and
                {pretty_label(strongest_pair[1])}
                show the strongest relationship in the current data
                (r = {strongest_r:.3f}).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 1.15]
    )

    with left:

        st.markdown(
            "### Descriptive statistics"
        )

        statistics = descriptive_stats(
            filtered_df[
                selected_variable
            ].tolist()
        )

        stats_table = pd.DataFrame(
            {
                "Statistic": [
                    "Mean",
                    "Median",
                    "Mode",
                    "Standard deviation",
                    "Variance",
                    "Range",
                    "IQR",
                    "Skewness",
                    "Kurtosis"
                ],
                "Value": [
                    statistics["mean"],
                    statistics["median"],
                    statistics["mode"],
                    statistics["std"],
                    statistics["variance"],
                    statistics["range"],
                    statistics["iqr"],
                    statistics["skewness"],
                    statistics["kurtosis"]
                ]
            }
        )

        stats_table[
            "Value"
        ] = stats_table[
            "Value"
        ].round(3)

        st.dataframe(
            stats_table,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown(
            "### Species profile"
        )

        species_summary = (
            filtered_df
            .groupby(
                "species"
            )[numeric_columns]
            .mean()
            .round(3)
        )

        species_summary.index = (
            species_summary
            .index
            .str.title()
        )

        species_summary.columns = [
            pretty_label(column)
            for column in species_summary.columns
        ]

        st.dataframe(
            species_summary,
            use_container_width=True
        )


# =================================================
# DISTRIBUTIONS
# =================================================

with tab_distribution:

    st.markdown(
        '<div class="section-label">Distribution profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="section-copy">
            Examine the shape and normality of
            {pretty_label(selected_variable).lower()}.
        </div>
        """,
        unsafe_allow_html=True
    )

    chart_left, chart_right = (
        st.columns(2)
    )

    with chart_left:

        histogram = (
            plot_histogram_with_distribution(
                filtered_df,
                selected_variable,
                distribution="normal"
            )
        )

        st.pyplot(
            histogram,
            use_container_width=True
        )

    with chart_right:

        qq_plot = (
            plot_qq_comparison(
                filtered_df,
                selected_variable
            )
        )

        st.pyplot(
            qq_plot,
            use_container_width=True
        )

    distribution_result = (
        fit_distribution(
            filtered_df[
                selected_variable
            ].tolist(),
            "normal"
        )
    )

    dist1, dist2, dist3, dist4 = (
        st.columns(4)
    )

    dist1.metric(
        "Fitted mean",
        f"{distribution_result['params'][0]:.3f}"
    )

    dist2.metric(
        "Fitted SD",
        f"{distribution_result['params'][1]:.3f}"
    )

    dist3.metric(
        "KS statistic",
        f"{distribution_result['ks_statistic']:.4f}"
    )

    dist4.metric(
        "P-value",
        format_p_value(
            distribution_result[
                "p_value"
            ]
        )
    )

    if distribution_result[
        "p_value"
    ] > 0.05:

        st.info(
            "The current data does not provide strong evidence "
            "against a normal distribution at α = 0.05."
        )

    else:

        st.warning(
            "The current data shows evidence of departure "
            "from normality at α = 0.05."
        )

    st.divider()

    st.markdown(
        "### Species comparison"
    )

    boxplot = (
        plot_boxplots_by_category(
            filtered_df,
            selected_variable,
            "species"
        )
    )

    st.pyplot(
        boxplot,
        use_container_width=True
    )


# =================================================
# RELATIONSHIPS
# =================================================

with tab_relationships:

    st.markdown(
        '<div class="section-label">Relationships</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-copy">
            Explore how the numerical flower measurements move together.
        </div>
        """,
        unsafe_allow_html=True
    )

    scatter = (
        create_interactive_scatter(
            filtered_df,
            scatter_x,
            scatter_y,
            "species"
        )
    )

    st.plotly_chart(
        scatter,
        use_container_width=True
    )

    st.divider()

    heatmap = (
        create_correlation_heatmap(
            filtered_df,
            numeric_columns
        )
    )

    st.pyplot(
        heatmap,
        use_container_width=True
    )

    with st.expander(
        "Correlation values"
    ):

        correlation_table = (
            filtered_df[
                numeric_columns
            ]
            .corr()
            .round(3)
        )

        correlation_table.index = [
            pretty_label(index)
            for index in correlation_table.index
        ]

        correlation_table.columns = [
            pretty_label(column)
            for column in correlation_table.columns
        ]

        st.dataframe(
            correlation_table,
            use_container_width=True
        )


# =================================================
# TESTS & INFERENCE
# =================================================

with tab_inference:

    st.markdown(
        '<div class="section-label">Estimation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-copy">
            Confidence intervals and hypothesis tests provide
            inferential evidence beyond the descriptive summaries.
        </div>
        """,
        unsafe_allow_html=True
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
        "Sample mean",
        f"{filtered_df[selected_variable].mean():.3f} cm"
    )

    ci2.metric(
        "95% CI · lower",
        f"{ci_lower:.3f} cm"
    )

    ci3.metric(
        "95% CI · upper",
        f"{ci_upper:.3f} cm"
    )

    st.divider()

    st.markdown(
        "### Two-species comparison"
    )

    st.caption(
        "Welch two-sample t-test for the selected measurement."
    )

    if len(
        analysis_species
    ) < 2:

        st.warning(
            "Select at least two species in the sidebar."
        )

    else:

        test_col1, test_col2 = (
            st.columns(2)
        )

        with test_col1:

            first_species = (
                st.selectbox(
                    "First species",
                    analysis_species,
                    index=0,
                    key="first_species",
                    format_func=lambda value: value.title()
                )
            )

        with test_col2:

            second_species = (
                st.selectbox(
                    "Second species",
                    analysis_species,
                    index=1,
                    key="second_species",
                    format_func=lambda value: value.title()
                )
            )

        if (
            first_species
            == second_species
        ):

            st.warning(
                "Choose two different species."
            )

        else:

            group1 = filtered_df[
                filtered_df[
                    "species"
                ] == first_species
            ][selected_variable].tolist()

            group2 = filtered_df[
                filtered_df[
                    "species"
                ] == second_species
            ][selected_variable].tolist()

            result = hypothesis_test(
                group1,
                group2,
                test_type="t-test"
            )

            test1, test2, test3 = (
                st.columns(3)
            )

            test1.metric(
                f"{first_species.title()} mean",
                f"{np.mean(group1):.3f} cm"
            )

            test2.metric(
                f"{second_species.title()} mean",
                f"{np.mean(group2):.3f} cm"
            )

            test3.metric(
                "P-value",
                format_p_value(
                    result["p_value"]
                )
            )

            st.caption(
                f"T-statistic: "
                f"{result['statistic']:.4f}"
            )

            if result[
                "p_value"
            ] < 0.05:

                st.success(
                    "Reject H₀ — the species differ significantly "
                    "for the selected measurement."
                )

            else:

                st.info(
                    "Fail to reject H₀ — no statistically significant "
                    "difference was detected."
                )

    st.divider()

    st.markdown(
        "### Across-species comparison"
    )

    st.caption(
        "One-way ANOVA for the species currently included."
    )

    if len(
        analysis_species
    ) < 2:

        st.warning(
            "Select at least two species in the sidebar."
        )

    else:

        groups = []

        for species in analysis_species:

            values = filtered_df[
                filtered_df[
                    "species"
                ] == species
            ][selected_variable].tolist()

            groups.append(
                values
            )

        result = anova_test(
            *groups
        )

        anova1, anova2 = (
            st.columns(2)
        )

        anova1.metric(
            "F-statistic",
            f"{result['f_statistic']:.3f}"
        )

        anova2.metric(
            "P-value",
            format_p_value(
                result[
                    "p_value"
                ]
            )
        )

        if result[
            "p_value"
        ] < 0.05:

            st.success(
                "Reject H₀ — at least one species mean differs significantly."
            )

        else:

            st.info(
                "Fail to reject H₀ — no significant mean difference was detected."
            )


# =================================================
# DATASET
# =================================================

with tab_data:

    st.markdown(
        '<div class="section-label">Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-copy">
            The cleaned observations currently included in the analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    csv_data = (
        filtered_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "Download filtered CSV",
        data=csv_data,
        file_name="iris_filtered_data.csv",
        mime="text/csv"
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Data Detective · Exploring Real-World Distributions"
)