import matplotlib.pyplot as plt
import pandas as pd
import pytest

from plotly.graph_objects import Figure as PlotlyFigure

from src.visualizations import (
    plot_histogram_with_distribution,
    create_correlation_heatmap,
    plot_boxplots_by_category,
    create_interactive_scatter,
    plot_qq_comparison,
    dashboard_layout
)


@pytest.fixture
def sample_data():
    """Small Iris-style dataset used for tests."""

    return pd.DataFrame({
        "sepal_length": [
            5.1, 4.9, 6.2, 6.5, 7.1, 6.3
        ],

        "sepal_width": [
            3.5, 3.0, 2.9, 3.0, 3.0, 3.3
        ],

        "petal_length": [
            1.4, 1.4, 4.3, 5.2, 5.9, 6.0
        ],

        "petal_width": [
            0.2, 0.2, 1.3, 2.0, 2.1, 2.5
        ],

        "species": [
            "setosa",
            "setosa",
            "versicolor",
            "versicolor",
            "virginica",
            "virginica"
        ]
    })


def test_histogram_with_distribution(sample_data):
    """Test histogram with fitted distribution."""

    fig = plot_histogram_with_distribution(
        sample_data,
        "petal_length"
    )

    assert isinstance(
        fig,
        plt.Figure
    )

    plt.close(fig)


def test_correlation_heatmap(sample_data):
    """Test correlation heatmap."""

    numerical_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    fig = create_correlation_heatmap(
        sample_data,
        numerical_columns
    )

    assert isinstance(
        fig,
        plt.Figure
    )

    plt.close(fig)


def test_boxplots_by_category(sample_data):
    """Test categorical boxplot."""

    fig = plot_boxplots_by_category(
        sample_data,
        "petal_length",
        "species"
    )

    assert isinstance(
        fig,
        plt.Figure
    )

    plt.close(fig)


def test_interactive_scatter(sample_data):
    """Test Plotly interactive scatter plot."""

    fig = create_interactive_scatter(
        sample_data,
        "petal_length",
        "petal_width",
        "species"
    )

    assert isinstance(
        fig,
        PlotlyFigure
    )


def test_qq_comparison(sample_data):
    """Test Q-Q plot."""

    fig = plot_qq_comparison(
        sample_data,
        "petal_length"
    )

    assert isinstance(
        fig,
        plt.Figure
    )

    plt.close(fig)


def test_dashboard_layout():
    """Test Streamlit dashboard configuration."""

    layout = dashboard_layout()

    assert layout["page_title"] == (
        "Iris Statistics Dashboard"
    )

    assert layout["layout"] == "wide"

    assert "page_icon" in layout


def test_invalid_column(sample_data):
    """Invalid columns should raise ValueError."""

    with pytest.raises(ValueError):

        plot_histogram_with_distribution(
            sample_data,
            "fake_column"
        )