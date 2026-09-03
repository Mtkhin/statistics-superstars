import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from plotly.graph_objects import Figure as PlotlyFigure


# =================================================
# VISUAL IDENTITY
# =================================================

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


# =================================================
# GLOBAL MATPLOTLIB DARK THEME
# =================================================

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": BACKGROUND,

    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,

    "xtick.color": MUTED,
    "ytick.color": MUTED,

    "axes.edgecolor": BORDER,
    "grid.color": BORDER,

    "font.family": "DejaVu Sans"
})


# =================================================
# HELPERS
# =================================================

def _check_column(
    data: pd.DataFrame,
    column: str
):
    """Check whether a column exists."""

    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' not found in dataset."
        )


def _pretty_label(
    value: str
) -> str:
    """Convert snake_case into a readable label."""

    return (
        value
        .replace("_", " ")
        .title()
    )


def _set_title(
    ax,
    text: str,
    fontsize: int = 15,
    pad: int = 14
):
    """
    Create a guaranteed light-colored left-aligned title.

    Matplotlib uses separate title objects for left,
    center, and right aligned titles.
    """

    # Remove any automatic centered title.
    ax.set_title(
        "",
        loc="center"
    )

    title = ax.set_title(
        text,
        loc="left",
        fontsize=fontsize,
        fontweight="semibold",
        color=TEXT,
        pad=pad
    )

    title.set_color(
        TEXT
    )

    return title


def _style_axes(
    ax,
    show_grid: bool = True
):
    """Apply dark styling to Matplotlib axes."""

    ax.set_facecolor(
        SURFACE
    )

    ax.tick_params(
        colors=MUTED,
        labelsize=10
    )

    ax.xaxis.label.set_color(
        MUTED
    )

    ax.yaxis.label.set_color(
        MUTED
    )

    # Style all possible Matplotlib title positions.
    ax.title.set_color(
        TEXT
    )

    ax._left_title.set_color(
        TEXT
    )

    ax._right_title.set_color(
        TEXT
    )

    for spine in ax.spines.values():

        spine.set_color(
            BORDER
        )

    if show_grid:

        ax.grid(
            color=BORDER,
            alpha=0.35,
            linewidth=0.6
        )

    else:

        ax.grid(
            False
        )


# =================================================
# HISTOGRAM + FITTED DISTRIBUTION
# =================================================

def plot_histogram_with_distribution(
    data: pd.DataFrame,
    column: str,
    bins: int = 15,
    distribution: str = "normal"
):
    """
    Plot a histogram with a fitted probability distribution.

    Supported:
        normal
        exponential

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
        linewidth=1,
        alpha=0.90,
        label="Observed data"
    )

    x_values = np.linspace(
        values.min(),
        values.max(),
        400
    )

    distribution = (
        distribution
        .lower()
    )

    if distribution == "normal":

        mean, std = stats.norm.fit(
            values
        )

        fitted_density = stats.norm.pdf(
            x_values,
            loc=mean,
            scale=std
        )

        distribution_label = (
            "Fitted normal"
        )

    elif distribution == "exponential":

        loc, scale = stats.expon.fit(
            values
        )

        fitted_density = stats.expon.pdf(
            x_values,
            loc=loc,
            scale=scale
        )

        distribution_label = (
            "Fitted exponential"
        )

    else:

        raise ValueError(
            "Unsupported distribution. "
            "Choose 'normal' or 'exponential'."
        )

    ax.plot(
        x_values,
        fitted_density,
        color=GOLD,
        linewidth=2.4,
        label=distribution_label
    )

    ax.set_xlabel(
        f"{_pretty_label(column)} (cm)"
    )

    ax.set_ylabel(
        "Density"
    )

    _style_axes(
        ax
    )

    _set_title(
        ax,
        f"Distribution of {_pretty_label(column)}"
    )

    legend = ax.legend(
        frameon=False,
        fontsize=9
    )

    for legend_text in legend.get_texts():

        legend_text.set_color(
            MUTED
        )

    fig.tight_layout()

    return fig


# =================================================
# CORRELATION HEATMAP
# =================================================

def create_correlation_heatmap(
    data: pd.DataFrame,
    numerical_columns: list
):
    """
    Create a dark-themed correlation heatmap.

    Returns:
        matplotlib Figure
    """

    for column in numerical_columns:

        _check_column(
            data,
            column
        )

    correlation_matrix = (
        data[
            numerical_columns
        ]
        .corr()
    )

    display_matrix = (
        correlation_matrix
        .copy()
    )

    display_matrix.index = [
        _pretty_label(value)
        for value
        in display_matrix.index
    ]

    display_matrix.columns = [
        _pretty_label(value)
        for value
        in display_matrix.columns
    ]

    custom_cmap = (
        LinearSegmentedColormap
        .from_list(
            "iris_correlation",
            [
                OXBLOOD,
                SURFACE,
                GOLD
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6.2),
        facecolor=BACKGROUND
    )

    heatmap = sns.heatmap(
        display_matrix,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=1,
        linecolor=BACKGROUND,
        cbar_kws={
            "shrink": 0.75,
            "pad": 0.03
        },
        annot_kws={
            "fontsize": 11
        },
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_facecolor(
        SURFACE
    )

    ax.tick_params(
        colors=MUTED,
        labelsize=10
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0
    )

    _set_title(
        ax,
        "Correlation Between Measurements",
        fontsize=16,
        pad=18
    )

    # Improve annotation visibility.
    for text_item in ax.texts:

        try:

            value = float(
                text_item.get_text()
            )

        except ValueError:

            continue

        if value >= 0.55:

            text_item.set_color(
                BACKGROUND
            )

        else:

            text_item.set_color(
                TEXT
            )

        if abs(value) >= 0.80:

            text_item.set_fontweight(
                "bold"
            )

    # Style colorbar.
    colorbar = (
        heatmap
        .collections[0]
        .colorbar
    )

    colorbar.ax.tick_params(
        colors=MUTED,
        labelsize=9
    )

    colorbar.outline.set_edgecolor(
        BORDER
    )

    fig.tight_layout()

    return fig


# =================================================
# BOXPLOTS
# =================================================

def plot_boxplots_by_category(
    data: pd.DataFrame,
    numerical_column: str,
    category_column: str
):
    """
    Create boxplots grouped by category.

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

    plot_data = (
        data.copy()
    )

    plot_data[
        category_column
    ] = (
        plot_data[
            category_column
        ]
        .astype(str)
        .str
        .title()
    )

    palette = {
        "Setosa": GOLD,
        "Versicolor": FOREST,
        "Virginica": OXBLOOD
    }

    fig, ax = plt.subplots(
        figsize=(8.5, 5.3),
        facecolor=BACKGROUND
    )

    sns.boxplot(
        data=plot_data,
        x=category_column,
        y=numerical_column,
        hue=category_column,
        palette=palette,
        legend=False,
        width=0.55,
        linewidth=1.1,
        ax=ax
    )

    ax.set_xlabel(
        "Species"
    )

    ax.set_ylabel(
        f"{_pretty_label(numerical_column)} (cm)"
    )

    _style_axes(
        ax
    )

    ax.grid(
        axis="x",
        visible=False
    )

    _set_title(
        ax,
        f"{_pretty_label(numerical_column)} Across Species"
    )

    fig.tight_layout()

    return fig


# =================================================
# INTERACTIVE SCATTER
# =================================================

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
                f"{_pretty_label(x_column)} "
                f"vs {_pretty_label(y_column)}"
            ),
            "x": 0.01,
            "xanchor": "left",
            "font": {
                "size": 20,
                "color": TEXT
            }
        },

        paper_bgcolor=BACKGROUND,
        plot_bgcolor=SURFACE,

        font={
            "color": TEXT
        },

        xaxis={
            "title": (
                f"{_pretty_label(x_column)} (cm)"
            ),
            "gridcolor": BORDER,
            "zerolinecolor": BORDER,
            "tickfont": {
                "color": MUTED
            },
            "title_font": {
                "color": MUTED
            }
        },

        yaxis={
            "title": (
                f"{_pretty_label(y_column)} (cm)"
            ),
            "gridcolor": BORDER,
            "zerolinecolor": BORDER,
            "tickfont": {
                "color": MUTED
            },
            "title_font": {
                "color": MUTED
            }
        },

        legend={
            "title": {
                "text": "Species"
            },
            "font": {
                "color": MUTED
            }
        },

        hoverlabel={
            "bgcolor": SURFACE_LIGHT,
            "font_color": TEXT,
            "bordercolor": BORDER
        },

        margin={
            "l": 25,
            "r": 25,
            "t": 65,
            "b": 25
        },

        height=520
    )

    return fig


# =================================================
# Q-Q PLOT
# =================================================

def plot_qq_comparison(
    data: pd.DataFrame,
    column: str
):
    """
    Create a Q-Q plot against a theoretical normal distribution.

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
        figsize=(7.5, 5),
        facecolor=BACKGROUND
    )

    stats.probplot(
        values,
        dist="norm",
        plot=ax
    )

    # IMPORTANT:
    # scipy automatically creates "Probability Plot".
    # Remove that title completely.
    ax.set_title(
        "",
        loc="center"
    )

    lines = (
        ax.get_lines()
    )

    if len(lines) >= 1:

        lines[0].set_markerfacecolor(
            FOREST
        )

        lines[0].set_markeredgecolor(
            FOREST
        )

        lines[0].set_markersize(
            5
        )

        lines[0].set_alpha(
            0.82
        )

    if len(lines) >= 2:

        lines[1].set_color(
            GOLD
        )

        lines[1].set_linewidth(
            2.2
        )

    ax.set_xlabel(
        "Theoretical Quantiles"
    )

    ax.set_ylabel(
        "Ordered Values"
    )

    _style_axes(
        ax
    )

    _set_title(
        ax,
        f"Q-Q Plot · {_pretty_label(column)}"
    )

    fig.tight_layout()

    return fig


# =================================================
# DASHBOARD CONFIGURATION
# =================================================

def dashboard_layout() -> dict:
    """Return Streamlit configuration."""

    return {
        "page_title": "Iris Statistical Study",
        "page_icon": "◼",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }


# =================================================
# BACKWARD COMPATIBILITY
# =================================================

def plot_histogram(
    data: pd.DataFrame,
    column: str,
    bins: int = 15
):
    """Backward-compatible histogram."""

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
    """Backward-compatible boxplot."""

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
    """Backward-compatible Matplotlib scatter."""

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

    ax.set_xlabel(
        _pretty_label(
            x_column
        )
    )

    ax.set_ylabel(
        _pretty_label(
            y_column
        )
    )

    _style_axes(
        ax
    )

    _set_title(
        ax,
        (
            f"{_pretty_label(x_column)} "
            f"vs {_pretty_label(y_column)}"
        )
    )

    legend = (
        ax.get_legend()
    )

    if legend:

        legend.get_frame().set_alpha(
            0
        )

        for text_item in legend.get_texts():

            text_item.set_color(
                MUTED
            )

    fig.tight_layout()

    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    numerical_columns: list
):
    """Backward-compatible heatmap."""

    return create_correlation_heatmap(
        data,
        numerical_columns
    )


def plot_qq(
    data: pd.DataFrame,
    column: str
):
    """Backward-compatible Q-Q plot."""

    return plot_qq_comparison(
        data,
        column
    )