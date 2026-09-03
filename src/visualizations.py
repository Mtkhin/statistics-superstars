import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from plotly.graph_objects import Figure as PlotlyFigure


# -------------------------------------------------
# Visual identity
# -------------------------------------------------

BACKGROUND = "#0E1110"
SURFACE = "#171B18"
SURFACE_LIGHT = "#202520"

TEXT = "#EEEAE2"
MUTED = "#A4AAA3"
BORDER = "#30352F"

GOLD = "#C4A66A"
FOREST = "#71806A"
OXBLOOD = "#8A5658"

SPECIES_PALETTE = {
    "setosa": GOLD,
    "versicolor": FOREST,
    "virginica": OXBLOOD
}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _check_column(data: pd.DataFrame, column: str):
    """Check whether a column exists."""

    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset."
        )


def _style_axes(ax):
    """Apply project styling to Matplotlib axes."""

    ax.set_facecolor(SURFACE)

    ax.tick_params(
        colors=MUTED
    )

    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

    for spine in ax.spines.values():
        spine.set_color(BORDER)

    ax.grid(
        alpha=0.12,
        color=MUTED
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
    Plot a histogram with a fitted distribution.

    Supported distributions:
        - normal
        - exponential

    Returns:
        matplotlib Figure
    """

    _check_column(data, column)

    values = (
        data[column]
        .dropna()
        .to_numpy()
    )

    if len(values) == 0:
        raise ValueError(
            "No valid data available for plotting."
        )

    fig, ax = plt.subplots(
        figsize=(8, 5),
        facecolor=BACKGROUND
    )

    ax.hist(
        values,
        bins=bins,
        density=True,
        color=FOREST,
        edgecolor=BACKGROUND,
        alpha=0.88,
        label="Observed data"
    )

    x_values = np.linspace(
        values.min(),
        values.max(),
        300
    )

    distribution = distribution.lower()

    if distribution == "normal":

        mean, std = stats.norm.fit(
            values
        )

        fitted_density = stats.norm.pdf(
            x_values,
            loc=mean,
            scale=std
        )

        label = "Fitted normal"

    elif distribution == "exponential":

        loc, scale = stats.expon.fit(
            values
        )

        fitted_density = stats.expon.pdf(
            x_values,
            loc=loc,
            scale=scale
        )

        label = "Fitted exponential"

    else:

        raise ValueError(
            "Unsupported distribution. "
            "Choose 'normal' or 'exponential'."
        )

    ax.plot(
        x_values,
        fitted_density,
        color=GOLD,
        linewidth=2.2,
        label=label
    )

    ax.set_title(
        f"Distribution of "
        f"{column.replace('_', ' ').title()}",
        loc="left",
        fontweight="bold"
    )

    ax.set_xlabel(
        f"{column.replace('_', ' ').title()} (cm)"
    )

    ax.set_ylabel(
        "Density"
    )

    _style_axes(ax)

    legend = ax.legend(
        frameon=False
    )

    for text in legend.get_texts():
        text.set_color(MUTED)

    fig.tight_layout()

    return fig


def create_correlation_heatmap(
    data: pd.DataFrame,
    numerical_columns: list
):
    """
    Create a styled correlation heatmap.

    Returns:
        matplotlib Figure
    """

    for column in numerical_columns:
        _check_column(
            data,
            column
        )

    correlation_matrix = (
        data[numerical_columns]
        .corr()
    )

    custom_cmap = LinearSegmentedColormap.from_list(
        "iris_correlation",
        [
            OXBLOOD,
            SURFACE,
            GOLD
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 6),
        facecolor=BACKGROUND
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        center=0,
        linewidths=0.8,
        linecolor=BACKGROUND,
        cbar_kws={
            "shrink": 0.8
        },
        ax=ax
    )

    ax.set_title(
        "Correlation Between Measurements",
        loc="left",
        fontweight="bold"
    )

    ax.tick_params(
        colors=MUTED
    )

    for label in ax.get_xticklabels():
        label.set_color(MUTED)

    for label in ax.get_yticklabels():
        label.set_color(MUTED)

    for text in ax.texts:
        text.set_color(TEXT)

    ax.set_facecolor(
        SURFACE
    )

    fig.tight_layout()

    return fig


def plot_boxplots_by_category(
    data: pd.DataFrame,
    numerical_column: str,
    category_column: str
):
    """
    Create a categorical boxplot.

    Returns:
        matplotlib Figure
    """

    _check_column(
        data,
        numerical_column
    )

    _check_column(
        data,
        category_column
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
        facecolor=BACKGROUND
    )

    available_categories = (
        data[category_column]
        .dropna()
        .unique()
    )

    palette = {
        category: SPECIES_PALETTE.get(
            str(category).lower(),
            GOLD
        )
        for category in available_categories
    }

    sns.boxplot(
        data=data,
        x=category_column,
        y=numerical_column,
        palette=palette,
        ax=ax
    )

    ax.set_title(
        f"{numerical_column.replace('_', ' ').title()} "
        "Across Species",
        loc="left",
        fontweight="bold"
    )

    ax.set_xlabel(
        "Species"
    )

    ax.set_ylabel(
        f"{numerical_column.replace('_', ' ').title()} (cm)"
    )

    _style_axes(ax)

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

    _check_column(
        data,
        x_column
    )

    _check_column(
        data,
        y_column
    )

    if category_column is not None:
        _check_column(
            data,
            category_column
        )

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=category_column,
        color_discrete_map=SPECIES_PALETTE,
        hover_data=data.columns.tolist()
    )

    fig.update_traces(
        marker={
            "size": 9,
            "opacity": 0.82,
            "line": {
                "width": 0.8,
                "color": BACKGROUND
            }
        }
    )

    fig.update_layout(
        title={
            "text": (
                f"{x_column.replace('_', ' ').title()} "
                f"vs {y_column.replace('_', ' ').title()}"
            ),
            "x": 0,
            "xanchor": "left"
        },
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=SURFACE,
        font={
            "color": TEXT
        },
        xaxis={
            "title": (
                f"{x_column.replace('_', ' ').title()} (cm)"
            ),
            "gridcolor": BORDER,
            "zerolinecolor": BORDER
        },
        yaxis={
            "title": (
                f"{y_column.replace('_', ' ').title()} (cm)"
            ),
            "gridcolor": BORDER,
            "zerolinecolor": BORDER
        },
        legend={
            "title": {
                "text": "Species"
            }
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 55,
            "b": 20
        }
    )

    return fig


def plot_qq_comparison(
    data: pd.DataFrame,
    column: str
):
    """
    Create a styled Q-Q plot.

    Returns:
        matplotlib Figure
    """

    _check_column(
        data,
        column
    )

    values = (
        data[column]
        .dropna()
    )

    if len(values) == 0:
        raise ValueError(
            "No valid data available for plotting."
        )

    fig, ax = plt.subplots(
        figsize=(7, 5),
        facecolor=BACKGROUND
    )

    stats.probplot(
        values,
        dist="norm",
        plot=ax
    )

    lines = ax.get_lines()

    if len(lines) >= 1:
        lines[0].set_markerfacecolor(
            FOREST
        )
        lines[0].set_markeredgecolor(
            FOREST
        )
        lines[0].set_alpha(
            0.8
        )

    if len(lines) >= 2:
        lines[1].set_color(
            GOLD
        )
        lines[1].set_linewidth(
            2
        )

    ax.set_title(
        f"Q-Q Plot · "
        f"{column.replace('_', ' ').title()}",
        loc="left",
        fontweight="bold"
    )

    _style_axes(ax)

    fig.tight_layout()

    return fig


def dashboard_layout() -> dict:
    """
    Return Streamlit page configuration.
    """

    return {
        "page_title": "Iris Statistical Study",
        "page_icon": "◼",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }


# -------------------------------------------------
# Backward-compatible functions
# -------------------------------------------------

def plot_histogram(
    data: pd.DataFrame,
    column: str,
    bins: int = 15
):
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
    Matplotlib scatter retained for notebook compatibility.
    """

    _check_column(
        data,
        x_column
    )

    _check_column(
        data,
        y_column
    )

    if group_column is not None:
        _check_column(
            data,
            group_column
        )

    fig, ax = plt.subplots(
        figsize=(8, 5),
        facecolor=BACKGROUND
    )

    if group_column is None:

        ax.scatter(
            data[x_column],
            data[y_column],
            color=GOLD,
            alpha=0.75
        )

    else:

        sns.scatterplot(
            data=data,
            x=x_column,
            y=y_column,
            hue=group_column,
            palette=SPECIES_PALETTE,
            ax=ax
        )

    ax.set_title(
        f"{x_column.replace('_', ' ').title()} "
        f"vs {y_column.replace('_', ' ').title()}",
        loc="left",
        fontweight="bold"
    )

    ax.set_xlabel(
        x_column.replace("_", " ").title()
    )

    ax.set_ylabel(
        y_column.replace("_", " ").title()
    )

    _style_axes(ax)

    fig.tight_layout()

    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    numerical_columns: list
):
    return create_correlation_heatmap(
        data,
        numerical_columns
    )


def plot_qq(
    data: pd.DataFrame,
    column: str
):
    return plot_qq_comparison(
        data,
        column
    )