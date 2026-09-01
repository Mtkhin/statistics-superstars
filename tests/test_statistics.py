import numpy as np
import pytest

from src.statistics import (
    descriptive_stats,
    fit_distribution,
    bootstrap_ci,
    hypothesis_test,
    anova_test
)

def test_descriptive_stats():
    """Test descriptive statistics calculation."""

    data = [1, 2, 2, 3, 4]

    result = descriptive_stats(data)

    assert result["mean"] == pytest.approx(2.4)
    assert result["median"] == 2.0
    assert result["mode"] == 2.0
    assert result["variance"] == pytest.approx(1.3)
    assert result["std"] == pytest.approx(1.140175, rel=1e-5)
    assert result["range"] == 3.0
    assert result["iqr"] == 1.0

    assert "skewness" in result
    assert "kurtosis" in result

def test_fit_normal_distribution():
    """Test fitting a normal distribution."""

    data = [1, 2, 3, 4, 5]

    result = fit_distribution(data, "normal")

    assert result["params"][0] == pytest.approx(3.0)
    assert result["params"][1] == pytest.approx(1.414214, rel=1e-5)

    assert 0 <= result["ks_statistic"] <= 1
    assert 0 <= result["p_value"] <= 1

def test_bootstrap_ci():
    """Test bootstrap confidence interval."""

    data = [4, 5, 5, 5, 6, 6, 7]

    lower, upper = bootstrap_ci(
        data,
        np.mean,
        n_bootstrap=2000,
        confidence=0.95
    )

    assert lower < upper
    assert lower <= np.mean(data) <= upper

def test_hypothesis_test_ttest():
    """Test independent two-sample t-test."""

    group1 = [5.1, 5.3, 5.0, 5.2, 5.4]
    group2 = [6.1, 6.3, 6.0, 6.2, 6.4]

    result = hypothesis_test(
        group1,
        group2,
        test_type="t-test"
    )

    assert result["test_type"] == "t-test"
    assert "statistic" in result
    assert "p_value" in result

    assert 0 <= result["p_value"] <= 1
    assert result["p_value"] < 0.05

def test_anova_test():
    """Test one-way ANOVA with three groups."""

    group1 = [1.0, 1.2, 1.1, 1.3, 1.2]
    group2 = [3.0, 3.2, 3.1, 3.3, 3.2]
    group3 = [5.0, 5.2, 5.1, 5.3, 5.2]

    result = anova_test(
        group1,
        group2,
        group3
    )

    assert "f_statistic" in result
    assert "p_value" in result

    assert result["f_statistic"] >= 0
    assert 0 <= result["p_value"] <= 1

    assert result["p_value"] < 0.05