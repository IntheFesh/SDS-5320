"""
This module implements a suite of causal inference estimators for the LaLonde
job‑training study.  The code is designed to accompany a comparative study of
observational estimators against an experimental benchmark, as described in
LaLonde (1986) and the follow‑up analyses of Dehejia and Wahba (1999, 2002).

The functions defined here are meant to be flexible: they allow the user to
load the Dehejia–Wahba version of the National Supported Work (NSW) data,
combine it with non‑experimental comparison groups, estimate a variety of
standard treatment effect estimators targeting the average treatment effect
on the treated (ATT), and compute simple covariate balance diagnostics.

By construction, none of the functions assume that the underlying data were
obtained from a randomized experiment.  Instead, they rely on different
adjustment strategies—regression adjustment, propensity score stratification,
nearest‑neighbour matching, inverse probability weighting, and augmented
inverse probability weighting—to recover the experimental benchmark when
applied to observational samples.

The code is written in pure Python and uses common scientific libraries
(``pandas``, ``numpy``, ``sklearn``, and ``statsmodels``).  To run the
examples, install these dependencies in your environment.  Note that the
module only defines functions; running it as a script will perform a simple
demonstration on the NSW experimental data if the required files are
available locally.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


_EPS = 1e-8


def _validate_treatment_column(df: pd.DataFrame, treat_col: str) -> None:
    """Validate that the treatment indicator exists and is binary (0/1)."""
    if treat_col not in df.columns:
        raise KeyError(f"Treatment column '{treat_col}' is missing from the input DataFrame.")
    treat_values = df[treat_col].dropna().unique()
    if not set(treat_values).issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{treat_col}' must be binary with values in {{0, 1}}; "
            f"found values {sorted(map(int, treat_values))}."
        )



def _validate_ps_array(
    df: pd.DataFrame,
    ps: np.ndarray,
    treat_col: str,
    check_common_support: bool = True,
) -> np.ndarray:
    """Validate propensity scores and optionally warn on common support issues."""
    ps = np.asarray(ps, dtype=float)
    if ps.ndim != 1:
        raise ValueError("Propensity score input 'ps' must be a 1D array.")
    if len(ps) != len(df):
        raise ValueError(f"Length mismatch: len(ps)={len(ps)} but len(df)={len(df)}.")
    if not np.isfinite(ps).all():
        raise ValueError("Propensity scores must all be finite numeric values.")
    if np.any(ps <= 0.0) or np.any(ps >= 1.0):
        raise ValueError("Propensity scores must lie strictly within (0, 1).")

    if check_common_support:
        treated = df[treat_col] == 1
        control = df[treat_col] == 0
        if treated.sum() == 0 or control.sum() == 0:
            warnings.warn(
                "Common support cannot be assessed because one treatment group is empty.",
                RuntimeWarning,
            )
            return ps

        t_min, t_max = ps[treated].min(), ps[treated].max()
        c_min, c_max = ps[control].min(), ps[control].max()
        overlap_low = max(t_min, c_min)
        overlap_high = min(t_max, c_max)

        if overlap_low >= overlap_high:
            warnings.warn(
                "No common support in propensity scores: treated and control PS ranges do not overlap.",
                RuntimeWarning,
            )
        else:
            outside_treated = int(((ps[treated] < overlap_low) | (ps[treated] > overlap_high)).sum())
            outside_control = int(((ps[control] < overlap_low) | (ps[control] > overlap_high)).sum())
            if outside_treated > 0 or outside_control > 0:
                warnings.warn(
                    "Potential common-support violation: "
                    f"{outside_treated} treated and {outside_control} control observations "
                    "fall outside the overlap interval.",
                    RuntimeWarning,
                )
    return ps


def load_lalonde_dw(
    treated_path: str, control_path: str,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load the Dehejia–Wahba NSW experimental sample.

    The function reads two plain‑text files containing treated and control
    observations from the NSW demonstration and concatenates them into a
    single ``DataFrame``.  Each line of the input files must contain the
    treatment indicator followed by the same ordered list of covariates and
    the outcome.  The default set of column names corresponds to the
    conventions used in the causal inference literature.

    Parameters
    ----------
    treated_path : str
        File system path to the ``nswre74_treated.txt`` file.  Each record in
        this file represents a programme participant.
    control_path : str
        File system path to the ``nswre74_control.txt`` file.  Each record in
        this file represents a NSW control subject from the experimental
        sample.
    columns : Iterable[str], optional
        Names to assign to the columns in the order they appear in the data.
        If ``None``, a sensible default is used: ``("treat", "age", "educ",
        "black", "hispan", "married", "nodegree", "re74", "re75", "re78")``.

    Returns
    -------
    pandas.DataFrame
        A single data frame combining treated and control records.  The
        ``treat`` column is a numeric indicator equal to one for treated
        observations and zero for controls.

    Notes
    -----
    Reading files from the National Bureau of Economic Research (NBER)
    servers sometimes requires manual download because the site rejects
    automated requests.  If you encounter HTTP 403 errors when attempting
    to download the datasets, download the files manually using a web
    browser and supply their file paths here.
    """
    if columns is None:
        columns = [
            "treat",
            "age",
            "educ",
            "black",
            "hispan",
            "married",
            "nodegree",
            "re74",
            "re75",
            "re78",
        ]
    # Read the treated and control files; whitespace separated
    treated_df = pd.read_csv(
        treated_path,
        sep="\s+",
        header=None,
        names=columns,
    )
    control_df = pd.read_csv(
        control_path,
        sep="\s+",
        header=None,
        names=columns,
    )
    df = pd.concat([treated_df, control_df], ignore_index=True)
    # Ensure the treatment indicator is binary integer type
    df["treat"] = df["treat"].astype(int)
    return df


def difference_in_means(
    df: pd.DataFrame,
    outcome: str = "re78",
    treat_col: str = "treat",
) -> Tuple[float, float, float]:
    """Compute the unadjusted difference in mean outcomes between treated and controls.

    This simple estimator serves as a baseline.  It does not account for
    confounding and therefore generally produces biased estimates when the
    treatment is not randomly assigned.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined sample of treated and control units.
    outcome : str, default "re78"
        Name of the outcome column.
    treat_col : str, default "treat"
        Name of the treatment indicator column.

    Returns
    -------
    att : float
        Estimated average treatment effect on the treated (difference in means).
    mean_treated : float
        Mean of the outcome among treated subjects.
    mean_control : float
        Mean of the outcome among control subjects.
    """
    _validate_treatment_column(df, treat_col)
    treated = df[df[treat_col] == 1][outcome]
    control = df[df[treat_col] == 0][outcome]
    mean_treated = treated.mean()
    mean_control = control.mean()
    att = mean_treated - mean_control
    return att, mean_treated, mean_control


def ols_regression_att(
    df: pd.DataFrame,
    outcome: str = "re78",
    treat_col: str = "treat",
    covariates: Optional[Iterable[str]] = None,
) -> float:
    """Estimate the treatment effect via linear regression adjustment.

    A linear outcome model is fitted to the full sample, regressing the
    outcome on the treatment indicator and a set of pre‑treatment covariates.
    The coefficient on the treatment variable is reported as the OLS
    estimate.  While this estimator is straightforward to compute, it
    implicitly assumes that the treatment effect is constant across
    individuals and that the outcome model is correctly specified.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the outcome, treatment indicator and covariates.
    outcome : str, default "re78"
        Outcome variable name.
    treat_col : str, default "treat"
        Treatment indicator column name.
    covariates : Iterable[str], optional
        List of covariate names to adjust for.  If ``None``, all columns
        except the treatment and outcome are used.

    Returns
    -------
    att_ols : float
        Coefficient on the treatment indicator from the linear regression.
    """
    _validate_treatment_column(df, treat_col)
    if covariates is None:
        covariates = [c for c in df.columns if c not in (outcome, treat_col)]
    # Construct the design matrix with an intercept
    X = df[[treat_col] + list(covariates)]
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome]
    # Fit ordinary least squares
    model = sm.OLS(y, X).fit()
    # Extract the coefficient on the treatment indicator
    att_ols = model.params[treat_col]
    return float(att_ols)


def estimate_propensity_score(
    df: pd.DataFrame,
    treat_col: str = "treat",
    covariates: Optional[Iterable[str]] = None,
    **logit_kwargs,
) -> np.ndarray:
    """Estimate propensity scores using logistic regression.

    A logistic regression classifier predicts the probability that each unit
    receives the treatment conditional on observed covariates.  The
    implementation uses scikit‑learn's ``LogisticRegression`` class with
    sensible defaults; additional keyword arguments are passed through to the
    constructor.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the treatment indicator and covariates.
    treat_col : str, default "treat"
        Name of the treatment indicator column.
    covariates : Iterable[str], optional
        List of covariate names used to predict the treatment.  If ``None``,
        all columns except the treatment indicator and outcome column named
        ``"re78"`` are used.  In particular, pre‑treatment earnings
        ``re74`` and ``re75`` are retained by default.
    **logit_kwargs
        Additional keyword arguments passed to ``LogisticRegression``.

    Returns
    -------
    ps : ndarray of shape (n_samples,)
        Estimated propensity scores for each unit.
    """
    _validate_treatment_column(df, treat_col)
    if covariates is None:
        # Exclude treatment and outcome only; keep pre-treatment covariates including re74/re75.
        exclude = {treat_col, "re78"}
        covariates = [c for c in df.columns if c not in exclude]
    covariates = list(covariates)
    if len(covariates) == 0:
        raise ValueError("At least one covariate is required to estimate propensity scores.")

    X = df[covariates].copy()
    y = df[treat_col].astype(int)
    # Default settings: L2 penalty, balanced class weight mitigates extreme probabilities
    logit = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        **logit_kwargs,
    )
    logit.fit(X, y)
    ps = logit.predict_proba(X)[:, 1]
    _validate_ps_array(df, ps, treat_col, check_common_support=True)
    return ps


def propensity_score_stratification_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    n_strata: int = 5,
    outcome: str = "re78",
    treat_col: str = "treat",
) -> float:
    """Compute the ATT using propensity score stratification.

    The sample is partitioned into ``n_strata`` strata based on quantiles of
    the propensity scores among the treated units.  Within each stratum the
    difference in mean outcomes between treated and control units is
    calculated.  These stratum‑specific effects are aggregated to produce the
    ATT, weighting each stratum by the proportion of treated subjects that
    fall into it.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample containing the treatment indicator and outcomes.
    ps : ndarray
        Pre‑computed propensity scores for all units.
    n_strata : int, default 5
        Number of strata (bins) to form.  Five strata correspond to
        quintiles, which is common in applied work.
    outcome : str, default "re78"
        Name of the outcome variable.
    treat_col : str, default "treat"
        Name of the treatment indicator column.

    Returns
    -------
    att_ps_strat : float
        Estimated ATT obtained via stratification on the propensity score.
    """
    _validate_treatment_column(df, treat_col)
    if n_strata < 2:
        raise ValueError("n_strata must be at least 2.")
    ps = _validate_ps_array(df, ps, treat_col, check_common_support=True)

    df = df.copy()
    df["ps"] = ps
    treated_ps = df.loc[df[treat_col] == 1, "ps"]
    quantiles = np.linspace(0, 1, n_strata + 1)
    bins = treated_ps.quantile(quantiles).to_numpy()
    bins = np.unique(bins)
    if len(bins) < 2:
        raise ValueError("Cannot form strata because treated propensity scores are nearly constant.")
    bins[0] -= _EPS
    bins[-1] += _EPS

    df["stratum"] = pd.cut(df["ps"], bins=bins, labels=False, include_lowest=True)

    att = 0.0
    n_treated = int((df[treat_col] == 1).sum())
    if n_treated == 0:
        return np.nan

    for stratum in sorted(df["stratum"].dropna().unique()):
        stratum_data = df[df["stratum"] == stratum]
        treated_group = stratum_data[stratum_data[treat_col] == 1]
        control_group = stratum_data[stratum_data[treat_col] == 0]
        if len(treated_group) == 0 or len(control_group) == 0:
            continue
        diff = treated_group[outcome].mean() - control_group[outcome].mean()
        weight = len(treated_group) / n_treated
        att += weight * diff
    return float(att)


def propensity_score_matching_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    outcome: str = "re78",
    treat_col: str = "treat",
    n_neighbors: int = 1,
    replace: bool = False,
) -> float:
    """Estimate the ATT via nearest‑neighbor propensity score matching.

    Each treated unit is paired with one or more control units whose propensity
    scores are closest in absolute distance.  The difference in outcomes
    between each treated unit and its matched controls is averaged across
    treated units to yield the ATT.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample containing the treatment indicator, propensity scores and
        outcomes.
    ps : ndarray
        Pre‑computed propensity scores for all units.
    outcome : str, default "re78"
        Name of the outcome variable.
    treat_col : str, default "treat"
        Name of the treatment indicator column.
    n_neighbors : int, default 1
        Number of control units to match to each treated unit.  Using more
        neighbours reduces variance at the expense of potential bias.
    replace : bool, default False
        Whether to match with replacement.  When ``True``, a control unit may
        be used as a match multiple times.  Matching without replacement
        attempts to find unique control matches for each treated unit; when
        the control sample is small relative to the treated sample, matching
        without replacement may produce poorer quality matches.

    Returns
    -------
    att_match : float
        Estimated ATT from nearest‑neighbour matching.
    """
    _validate_treatment_column(df, treat_col)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    ps = _validate_ps_array(df, ps, treat_col, check_common_support=True)

    df = df.copy()
    df["ps"] = ps
    treated_df = df[df[treat_col] == 1].reset_index(drop=True)
    control_df = df[df[treat_col] == 0].reset_index(drop=True)

    if len(control_df) == 0 or len(treated_df) == 0:
        return np.nan

    if replace and n_neighbors > len(control_df):
        raise ValueError("n_neighbors cannot exceed number of controls when matching with replacement.")

    treated_scores = treated_df["ps"].to_numpy().reshape(-1, 1)
    control_scores = control_df["ps"].to_numpy().reshape(-1, 1)

    if replace:
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(control_scores)
        _, indices = nn.kneighbors(treated_scores)
    else:
        # Query all controls so we can find unused neighbors sequentially.
        nn = NearestNeighbors(n_neighbors=len(control_df))
        nn.fit(control_scores)
        _, indices = nn.kneighbors(treated_scores)

    used_control_indices = set()
    effects = []
    for i, neigh_idx in enumerate(indices):
        if not replace:
            available = [idx for idx in neigh_idx if idx not in used_control_indices]
            if not available:
                continue
            match_idx = available[0]
            used_control_indices.add(match_idx)
            matched_control_outcomes = control_df.iloc[[match_idx]][outcome]
        else:
            matched_control_outcomes = control_df.iloc[neigh_idx][outcome]
        effect = treated_df.iloc[i][outcome] - matched_control_outcomes.mean()
        effects.append(effect)

    return float(np.mean(effects)) if effects else np.nan


def ipw_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    outcome: str = "re78",
    treat_col: str = "treat",
) -> float:
    """Compute the inverse probability weighting estimator for the ATT.

    This estimator re‑weights control observations so that the distribution of
    covariates among the weighted controls matches that of the treated group.
    The weights are defined as one for each treated unit and ``ps/(1-ps)`` for
    each control unit.  The ATT is the difference between the mean outcome of
    treated units and the weighted mean outcome of controls.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample containing the treatment indicator and outcome.
    ps : ndarray
        Propensity scores for each unit.
    outcome : str, default "re78"
        Name of the outcome variable.
    treat_col : str, default "treat"
        Name of the treatment indicator column.

    Returns
    -------
    att_ipw : float
        Estimated ATT using inverse probability weighting.
    """
    _validate_treatment_column(df, treat_col)
    ps = _validate_ps_array(df, ps, treat_col, check_common_support=True)

    df = df.copy()
    df["ps"] = ps
    weights = np.where(df[treat_col] == 1, 1.0, df["ps"] / (1.0 - df["ps"] + _EPS))
    df["w_ipw"] = weights

    treated_outcomes = df.loc[df[treat_col] == 1, outcome]
    control_outcomes = df.loc[df[treat_col] == 0, outcome]
    control_weights = df.loc[df[treat_col] == 0, "w_ipw"]

    if control_weights.sum() <= 0:
        raise ValueError("Control IPW weights must sum to a positive value.")

    mean_control = np.average(control_outcomes, weights=control_weights)
    mean_treated = treated_outcomes.mean()
    att_ipw = mean_treated - mean_control
    return float(att_ipw)


def aipw_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    outcome: str = "re78",
    treat_col: str = "treat",
    covariates: Optional[Iterable[str]] = None,
) -> float:
    """Estimate the ATT using augmented inverse probability weighting.

    The augmented IPW estimator combines outcome modeling with inverse
    probability weighting to achieve double robustness: it remains consistent
    if either the propensity score model or the outcome model is correctly
    specified.  A linear model for the control outcome regression ``mu0(x)``
    is fit and then combined with control reweighting by ``ps/(1-ps)``.  The
    implemented ATT estimator is the common doubly robust form

    ``ATT = (1 / n1) * sum_i[ D_i*(Y_i - mu0(X_i)) - (1-D_i)*e(X_i)/(1-e(X_i))*(Y_i - mu0(X_i)) ]``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the outcome, treatment indicator and covariates.
    ps : ndarray
        Propensity scores for each unit.
    outcome : str, default "re78"
        Name of the outcome variable.
    treat_col : str, default "treat"
        Name of the treatment indicator column.
    covariates : Iterable[str], optional
        List of covariate names used in the outcome model.  If ``None``,
        all columns except the treatment and outcome are used.

    Returns
    -------
    att_aipw : float
        Estimated ATT using augmented inverse probability weighting.
    """
    _validate_treatment_column(df, treat_col)
    ps = _validate_ps_array(df, ps, treat_col, check_common_support=True)

    if covariates is None:
        covariates = [c for c in df.columns if c not in (outcome, treat_col)]
    covariates = list(covariates)

    df = df.copy()
    X = sm.add_constant(df[covariates], has_constant="add")

    treated_mask = df[treat_col] == 1
    control_mask = ~treated_mask
    n1 = int(treated_mask.sum())
    if n1 == 0:
        return np.nan

    model_control = sm.OLS(df.loc[control_mask, outcome], X.loc[control_mask]).fit()
    mu0_hat = model_control.predict(X).to_numpy()

    D = df[treat_col].to_numpy(dtype=float)
    Y = df[outcome].to_numpy(dtype=float)
    resid0 = Y - mu0_hat
    control_weight = ps / (1.0 - ps + _EPS)

    terms = D * resid0 - (1.0 - D) * control_weight * resid0
    att = terms.sum() / n1
    return float(att)


def covariate_balance(
    df: pd.DataFrame,
    covariates: Iterable[str],
    treat_col: str = "treat",
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standardized mean differences (SMD) for covariates.

    Standardized differences compare covariate distributions between treated
    and control groups.  For each covariate the difference in means is
    divided by the pooled standard deviation.  When weights are provided,
    the weighted mean and variance are used for the control group; the
    treated group is always unweighted for ATT applications.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the covariates, treatment indicator and optionally
        weights.
    covariates : Iterable[str]
        Names of covariate columns for which balance is assessed.
    treat_col : str, default "treat"
        Name of the treatment indicator column.
    weights : ndarray, optional
        Weights applied to control observations.  If ``None``, ordinary
        unweighted means are used for both treated and controls.

    Returns
    -------
    smd : dict
        A mapping from covariate names to standardized mean differences.  A
        smaller absolute SMD indicates better covariate balance.
    """
    _validate_treatment_column(df, treat_col)
    smd = {}
    treated_mask = df[treat_col] == 1
    control_mask = df[treat_col] == 0
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(df):
            raise ValueError("Length of weights must match number of rows in df.")
        if np.any(weights < 0):
            raise ValueError("Weights must be nonnegative.")

    for cov in covariates:
        x_treated = df.loc[treated_mask, cov].astype(float)
        x_control = df.loc[control_mask, cov].astype(float)
        mean_t = x_treated.mean()
        if weights is None:
            mean_c = x_control.mean()
            var_t = x_treated.var(ddof=1)
            var_c = x_control.var(ddof=1)
        else:
            w_control = weights[control_mask]
            if w_control.sum() <= 0:
                raise ValueError("Control weights must sum to a positive value.")
            mean_c = np.average(x_control, weights=w_control)
            var_c = np.sum(w_control * (x_control - mean_c) ** 2) / w_control.sum()
            var_t = x_treated.var(ddof=1)
        pooled_std = np.sqrt((var_t + var_c) / 2.0)
        smd[cov] = (mean_t - mean_c) / (pooled_std + _EPS)
    return smd


def example_usage():
    """Demonstrate the estimators on NSW experimental files in the cwd.

    This function is intended for interactive exploration.  It attempts to
    load ``nswre74_treated.txt`` and ``nswre74_control.txt`` from the current
    working directory.  If found, it prints a compact ATT summary table and
    covariate-balance diagnostics (raw and IPW-weighted SMDs).
    """
    import os

    treated_path = os.path.join(os.getcwd(), "nswre74_treated.txt")
    control_path = os.path.join(os.getcwd(), "nswre74_control.txt")
    if not (os.path.exists(treated_path) and os.path.exists(control_path)):
        print(
            "Demo data files not found. Place 'nswre74_treated.txt' and 'nswre74_control.txt' in the working directory."
        )
        return

    df = load_lalonde_dw(treated_path, control_path)
    covariates = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]

    att_dm, mean_t, mean_c = difference_in_means(df)
    att_ols = ols_regression_att(df)
    ps = estimate_propensity_score(df, covariates=covariates)
    att_strat = propensity_score_stratification_att(df, ps)
    att_match = propensity_score_matching_att(df, ps, replace=True)
    att_ipw = ipw_att(df, ps)
    att_aipw = aipw_att(df, ps, covariates=covariates)

    summary_df = pd.DataFrame(
        [
            {"Estimator": "Difference in Means", "ATT": att_dm},
            {"Estimator": "OLS Regression", "ATT": att_ols},
            {"Estimator": "PS Stratification", "ATT": att_strat},
            {"Estimator": "PS Matching (NN, replacement)", "ATT": att_match},
            {"Estimator": "IPW (ATT)", "ATT": att_ipw},
            {"Estimator": "AIPW (ATT)", "ATT": att_aipw},
        ]
    )
    summary_df["ATT"] = summary_df["ATT"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "nan")

    print("\nOutcome means:")
    print(f"  Treated mean ({len(df[df['treat'] == 1])} units): {mean_t:,.2f}")
    print(f"  Control mean ({len(df[df['treat'] == 0])} units): {mean_c:,.2f}")
    print("\nATT summary table:")
    print(summary_df.to_string(index=False))

    bal_unweighted = covariate_balance(df, covariates)
    weights = np.where(df["treat"] == 1, 1.0, ps / (1.0 - ps + _EPS))
    bal_weighted = covariate_balance(df, covariates, weights=weights)

    balance_df = pd.DataFrame(
        {
            "Covariate": covariates,
            "SMD Unweighted": [bal_unweighted[c] for c in covariates],
            "SMD IPW-weighted": [bal_weighted[c] for c in covariates],
        }
    )
    for col in ["SMD Unweighted", "SMD IPW-weighted"]:
        balance_df[col] = balance_df[col].map(lambda x: f"{x:+.3f}")

    print("\nCovariate balance (standardized mean differences):")
    print(balance_df.to_string(index=False))


if __name__ == "__main__":
    # Run the demonstration when executed as a script
    example_usage()
