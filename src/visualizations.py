import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats
from plotly.graph_objects import Figure as PlotlyFigure


# -------------------------------------------------
# Helper
# -------------------------------------------------

def _check_column(data: pd.DataFrame, column: str):
    """Check whether a column exists in the dataset."""

    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset."
        )


# -------------------------------------------------
# Professor-required visualisation functions
# -------------------------------------------------

def plot_histogram_with_distribution(
    data: pd.DataFrame,
    column: str,
    bins: int = 15,
    distribution: str = "normal"
):
    """
    Plot a histogram with a fitted probability distribution.

    Supported distributions:
        - normal
        - exponential

    Returns:
        matplotlib Figure
    """

    _check_column(data, column)

    values = data[column].dropna().to_numpy()

    if len(values) == 0:
        raise ValueError("No valid data available for plotting.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.7,
        edgecolor="black",
        label="Observed data"
    )

    x_values = np.linspace(
        values.min(),
        values.max(),
        300
    )

    distribution = distribution.lower()

    if distribution == "normal":

        mean, std = stats.norm.fit(values)

        fitted_density = stats.norm.pdf(
            x_values,
            loc=mean,
            scale=std
        )

        distribution_label = "Fitted Normal Distribution"

    elif distribution == "exponential":

        loc, scale = stats.expon.fit(values)

        fitted_density = stats.expon.pdf(
            x_values,
            loc=loc,
            scale=scale
        )

        distribution_label = "Fitted Exponential Distribution"

    else:

        raise ValueError(
            "Unsupported distribution. "
            "Choose 'normal' or 'exponential'."
        )

    ax.plot(
        x_values,
        fitted_density,
        linewidth=2,
        label=distribution_label
    )

    ax.set_title(
        f"Distribution of "
        f"{column.replace('_', ' ').title()}"
    )

    ax.set_xlabel(
        column.replace("_", " ").title()
    )

    ax.set_ylabel("Density")

    ax.legend()

    fig.tight_layout()

    return fig


def create_correlation_heatmap(
    data: pd.DataFrame,
    numerical_columns: list
):
    """
    Create a correlation heatmap.

    Returns:
        matplotlib Figure
    """

    for column in numerical_columns:
        _check_column(data, column)

    correlation_matrix = (
        data[numerical_columns]
        .corr()
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix of Iris Measurements"
    )

    fig.tight_layout()

    return fig


def plot_boxplots_by_category(
    data: pd.DataFrame,
    numerical_column: str,
    category_column: str
):
    """
    Create a boxplot comparing a numerical variable
    across categories.

    Returns:
        matplotlib Figure
    """

    _check_column(data, numerical_column)
    _check_column(data, category_column)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=data,
        x=category_column,
        y=numerical_column,
        ax=ax
    )

    ax.set_title(
        f"{numerical_column.replace('_', ' ').title()} "
        f"by {category_column.replace('_', ' ').title()}"
    )

    ax.set_xlabel(
        category_column.replace("_", " ").title()
    )

    ax.set_ylabel(
        f"{numerical_column.replace('_', ' ').title()} (cm)"
    )

    fig.tight_layout()

    return fig


def create_interactive_scatter(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    category_column: str = None
) -> PlotlyFigure:
    """
    Create an interactive Plotly scatter plot.

    Returns:
        Plotly Figure
    """

    _check_column(data, x_column)
    _check_column(data, y_column)

    if category_column is not None:
        _check_column(data, category_column)

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=category_column,
        hover_data=data.columns.tolist(),
        title=(
            f"{x_column.replace('_', ' ').title()} "
            f"vs {y_column.replace('_', ' ').title()}"
        )
    )

    fig.update_layout(
        xaxis_title=(
            f"{x_column.replace('_', ' ').title()} (cm)"
        ),
        yaxis_title=(
            f"{y_column.replace('_', ' ').title()} (cm)"
        ),
        legend_title=(
            category_column.replace("_", " ").title()
            if category_column
            else None
        )
    )

    return fig


def plot_qq_comparison(
    data: pd.DataFrame,
    column: str
):
    """
    Create a Q-Q plot comparing observed data
    with a theoretical normal distribution.

    Returns:
        matplotlib Figure
    """

    _check_column(data, column)

    values = data[column].dropna()

    if len(values) == 0:
        raise ValueError("No valid data available for plotting.")

    fig, ax = plt.subplots(figsize=(7, 5))

    stats.probplot(
        values,
        dist="norm",
        plot=ax
    )

    ax.set_title(
        f"Q-Q Plot: "
        f"{column.replace('_', ' ').title()}"
    )

    fig.tight_layout()

    return fig


def dashboard_layout() -> dict:
    """
    Return Streamlit dashboard page configuration.
    """

    return {
        "page_title": "Iris Statistics Dashboard",
        "page_icon": "🌸",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }


# -------------------------------------------------
# Backward-compatible functions
# -------------------------------------------------
# These are kept because the earlier notebooks
# already use these function names.
# -------------------------------------------------

def plot_histogram(
    data: pd.DataFrame,
    column: str,
    bins: int = 15
):
    """Backward-compatible histogram function."""

    return plot_histogram_with_distribution(
        data=data,
        column=column,
        bins=bins,
        distribution="normal"
    )


def plot_boxplot(
    data: pd.DataFrame,
    numerical_column: str,
    group_column: str
):
    """Backward-compatible boxplot function."""

    return plot_boxplots_by_category(
        data=data,
        numerical_column=numerical_column,
        category_column=group_column
    )


def plot_scatter(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str = None
):
    """
    Backward-compatible Matplotlib scatter plot
    used by the existing notebook.
    """

    _check_column(data, x_column)
    _check_column(data, y_column)

    if group_column is not None:
        _check_column(data, group_column)

    fig, ax = plt.subplots(figsize=(8, 5))

    if group_column is None:

        ax.scatter(
            data[x_column],
            data[y_column],
            alpha=0.7
        )

    else:

        sns.scatterplot(
            data=data,
            x=x_column,
            y=y_column,
            hue=group_column,
            ax=ax
        )

    ax.set_title(
        f"{x_column.replace('_', ' ').title()} "
        f"vs {y_column.replace('_', ' ').title()}"
    )

    ax.set_xlabel(
        x_column.replace("_", " ").title()
    )

    ax.set_ylabel(
        y_column.replace("_", " ").title()
    )

    fig.tight_layout()

    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    numerical_columns: list
):
    """Backward-compatible heatmap function."""

    return create_correlation_heatmap(
        data,
        numerical_columns
    )


def plot_qq(
    data: pd.DataFrame,
    column: str
):
    """Backward-compatible Q-Q plot function."""

    return plot_qq_comparison(
        data,
        column
    )