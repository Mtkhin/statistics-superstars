import numpy as np
from scipy import stats
from typing import List, Tuple, Callable


def descriptive_stats(data: List[float]) -> dict:
    """
    Calculate key descriptive statistics.

    Returns:
        {
            'mean': float,
            'median': float,
            'mode': float,
            'std': float,
            'variance': float,
            'range': float,
            'iqr': float,
            'skewness': float,
            'kurtosis': float
        }
    """

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("Data cannot be empty.")

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    mode_result = stats.mode(data, keepdims=True)

    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "mode": float(mode_result.mode[0]),
        "std": float(np.std(data, ddof=1)),
        "variance": float(np.var(data, ddof=1)),
        "range": float(np.max(data) - np.min(data)),
        "iqr": float(q3 - q1),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data))
    }


def fit_distribution(data: List[float], dist_name: str) -> dict:
    """
    Fit a probability distribution to data.

    Supported distributions:
        - normal
        - exponential
        - poisson
        - binomial

    Returns:
        {
            'params': tuple,
            'ks_statistic': float,
            'p_value': float
        }
    """

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("Data cannot be empty.")

    dist_name = dist_name.lower()

    if dist_name == "normal":
        mean, std = stats.norm.fit(data)

        if std == 0:
            raise ValueError(
                "Normal distribution cannot be fitted to constant data."
            )

        ks_statistic, p_value = stats.kstest(
            data,
            lambda x: stats.norm.cdf(
                x,
                loc=mean,
                scale=std
            )
        )

        params = (mean, std)

    elif dist_name == "exponential":
        loc, scale = stats.expon.fit(data)

        if scale <= 0:
            raise ValueError(
                "Invalid scale parameter for exponential distribution."
            )

        ks_statistic, p_value = stats.kstest(
            data,
            lambda x: stats.expon.cdf(
                x,
                loc=loc,
                scale=scale
            )
        )

        params = (loc, scale)

    elif dist_name == "poisson":
        if np.any(data < 0) or not np.allclose(data, np.round(data)):
            raise ValueError(
                "Poisson distribution requires non-negative integer data."
            )

        lambda_value = np.mean(data)

        ks_statistic, p_value = stats.kstest(
            data,
            lambda x: stats.poisson.cdf(
                x,
                mu=lambda_value
            )
        )

        params = (lambda_value,)

    elif dist_name == "binomial":
        if np.any(data < 0) or not np.allclose(data, np.round(data)):
            raise ValueError(
                "Binomial distribution requires non-negative integer data."
            )

        n = int(np.max(data))

        if n <= 0:
            raise ValueError(
                "Binomial data must contain positive count values."
            )

        p = np.mean(data) / n
        p = min(max(p, 0.0), 1.0)

        ks_statistic, p_value = stats.kstest(
            data,
            lambda x: stats.binom.cdf(
                x,
                n=n,
                p=p
            )
        )

        params = (n, p)

    else:
        raise ValueError(
            "Unsupported distribution. Choose "
            "'normal', 'exponential', 'poisson', or 'binomial'."
        )

    return {
        "params": tuple(float(value) for value in params),
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value)
    }


def bootstrap_ci(
    data: List[float],
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute a bootstrap confidence interval.

    Args:
        data: Numerical observations.
        statistic: Function used to calculate a statistic,
                   such as np.mean or np.median.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.

    Returns:
        Tuple containing lower and upper confidence limits.
    """

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("Data cannot be empty.")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(42)

    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        sample = rng.choice(
            data,
            size=len(data),
            replace=True
        )

        bootstrap_statistics.append(
            statistic(sample)
        )

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_statistics,
        100 * (alpha / 2)
    )

    upper = np.percentile(
        bootstrap_statistics,
        100 * (1 - alpha / 2)
    )

    return float(lower), float(upper)


def hypothesis_test(
    data1: List[float],
    data2: List[float],
    test_type: str = "t-test"
) -> dict:
    """
    Perform a hypothesis test.

    Supported tests:
        - t-test
        - mann-whitney
        - chi-square

    Returns:
        {
            'test_type': str,
            'statistic': float,
            'p_value': float
        }
    """

    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)

    if data1.size == 0 or data2.size == 0:
        raise ValueError("Input data cannot be empty.")

    test_type = test_type.lower()

    if test_type == "t-test":
        statistic, p_value = stats.ttest_ind(
            data1,
            data2,
            equal_var=False
        )

    elif test_type == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(
            data1,
            data2,
            alternative="two-sided"
        )

    elif test_type == "chi-square":
        if len(data1) != len(data2):
            raise ValueError(
                "Chi-square input groups must have the same number of categories."
            )

        if np.any(data1 < 0) or np.any(data2 < 0):
            raise ValueError(
                "Chi-square frequencies cannot be negative."
            )

        contingency_table = np.array([
            data1,
            data2
        ])

        statistic, p_value, _, _ = stats.chi2_contingency(
            contingency_table
        )

    else:
        raise ValueError(
            "Unsupported test. Choose "
            "'t-test', 'mann-whitney', or 'chi-square'."
        )

    return {
        "test_type": test_type,
        "statistic": float(statistic),
        "p_value": float(p_value)
    }


def anova_test(*groups: List[float]) -> dict:
    """
    Perform a one-way ANOVA test.

    Returns:
        {
            'f_statistic': float,
            'p_value': float
        }
    """

    if len(groups) < 2:
        raise ValueError(
            "ANOVA requires at least two groups."
        )

    cleaned_groups = []

    for group in groups:
        group_array = np.asarray(group, dtype=float)

        if group_array.size == 0:
            raise ValueError(
                "ANOVA groups cannot be empty."
            )

        cleaned_groups.append(group_array)

    f_statistic, p_value = stats.f_oneway(
        *cleaned_groups
    )

    return {
        "f_statistic": float(f_statistic),
        "p_value": float(p_value)
    }