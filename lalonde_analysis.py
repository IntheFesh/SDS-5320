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

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from typing import Iterable, Optional, Tuple, Dict


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
    if covariates is None:
        covariates = [c for c in df.columns if c not in (outcome, treat_col)]
    # Construct the design matrix with an intercept
    X = df[[treat_col] + list(covariates)]
    X = sm.add_constant(X, has_constant='add')
    y = df[outcome]
    # Fit ordinary least squares
    model = sm.OLS(y, X).fit()
    # Extract the coefficient on the treatment indicator
    att_ols = model.params[treat_col]
    return att_ols


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
        all columns except the treatment indicator and any outcome column
        named ``"re78"`` are used.
    **logit_kwargs
        Additional keyword arguments passed to ``LogisticRegression``.

    Returns
    -------
    ps : ndarray of shape (n_samples,)
        Estimated propensity scores for each unit.
    """
    if covariates is None:
        # Exclude typical outcome columns if present
        exclude = {treat_col, 're78', 're75', 're74'}  # do not exclude re74/re75 for PS by default
        covariates = [c for c in df.columns if c not in exclude]
    X = df[list(covariates)].copy()
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
    df = df.copy()
    df["ps"] = ps
    # Determine strata boundaries based on treated units' PS distribution
    treated_ps = df.loc[df[treat_col] == 1, "ps"]
    quantiles = np.linspace(0, 1, n_strata + 1)
    bins = treated_ps.quantile(quantiles).values
    # Ensure the bins are strictly increasing to avoid issues in cut
    bins[0] = bins[0] - 1e-8
    bins[-1] = bins[-1] + 1e-8
    df["stratum"] = pd.cut(df["ps"], bins=bins, labels=False, include_lowest=True)
    att = 0.0
    # Total number of treated units for weighting
    n_treated = (df[treat_col] == 1).sum()
    for stratum in range(n_strata):
        stratum_data = df[df["stratum"] == stratum]
        treated_group = stratum_data[stratum_data[treat_col] == 1]
        control_group = stratum_data[stratum_data[treat_col] == 0]
        # Skip strata with no treated or no controls
        if len(treated_group) == 0 or len(control_group) == 0:
            continue
        # Stratum effect and weight by share of treated
        diff = treated_group[outcome].mean() - control_group[outcome].mean()
        weight = len(treated_group) / n_treated
        att += weight * diff
    return att


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
    df = df.copy()
    df["ps"] = ps
    # Separate treated and control units
    treated_df = df[df[treat_col] == 1].reset_index(drop=True)
    control_df = df[df[treat_col] == 0].reset_index(drop=True)
    # Prepare the feature arrays for matching
    treated_scores = treated_df["ps"].values.reshape(-1, 1)
    control_scores = control_df["ps"].values.reshape(-1, 1)
    # Fit nearest neighbour model on control scores
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(control_scores)
    distances, indices = nn.kneighbors(treated_scores)
    # Keep track of matched control indices if matching without replacement
    used_control_indices = set()
    effects = []
    for i, neigh_idx in enumerate(indices):
        # Optionally enforce unique matches
        if not replace:
            # Filter out already used controls and find the next nearest control
            available = [idx for idx in neigh_idx if idx not in used_control_indices]
            if not available:
                # If no available matches, skip this treated unit
                continue
            # Use the first available control as the match
            match_idx = available[0]
            used_control_indices.add(match_idx)
            matched_control_outcomes = control_df.iloc[[match_idx]][outcome]
        else:
            # With replacement, average over the specified neighbours
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
    df = df.copy()
    df["ps"] = ps
    # Compute weights: 1 for treated, ps/(1-ps) for controls
    weights = np.where(
        df[treat_col] == 1,
        1.0,
        df["ps"] / (1.0 - df["ps"] + 1e-8),  # add small constant to avoid division by zero
    )
    df["w_ipw"] = weights
    # Mean outcomes for treated and weighted controls
    treated_outcomes = df.loc[df[treat_col] == 1, outcome]
    control_outcomes = df.loc[df[treat_col] == 0, outcome]
    control_weights = df.loc[df[treat_col] == 0, "w_ipw"]
    # Weighted mean for controls
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
    specified.  Separate linear outcome models are fitted for treated and
    control units, predicting the outcome as a function of the covariates.
    These models are then used to impute missing potential outcomes.  The
    estimator here follows the formula for the ATT described in the causal
    inference literature.

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
        List of covariate names used in the outcome models.  If ``None``,
        all columns except the treatment and outcome are used.

    Returns
    -------
    att_aipw : float
        Estimated ATT using augmented inverse probability weighting.
    """
    if covariates is None:
        covariates = [c for c in df.columns if c not in (outcome, treat_col)]
    df = df.copy()
    df["ps"] = ps
    # Fit separate outcome models for treated and control units
    X = sm.add_constant(df[covariates], has_constant='add')
    # Outcome model for treated
    treated_mask = df[treat_col] == 1
    model_treated = sm.OLS(df.loc[treated_mask, outcome], X.loc[treated_mask]).fit()
    mu1_hat = model_treated.predict(X)
    # Outcome model for controls
    control_mask = df[treat_col] == 0
    model_control = sm.OLS(df.loc[control_mask, outcome], X.loc[control_mask]).fit()
    mu0_hat = model_control.predict(X)
    # Components of the AIPW estimator
    D = df[treat_col].values.astype(float)
    Y = df[outcome].values.astype(float)
    p = ps
    # Weights for controls in ATT setting
    weights = p / (1.0 - p + 1e-8)
    # AIPW influence function contribution for each observation
    n1 = treated_mask.sum()
    terms = np.where(
        D == 1,
        Y - mu0_hat,
        weights * (mu1_hat - Y),
    )
    att = terms.sum() / (n1 if n1 > 0 else 1)
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
    smd = {}
    treated_mask = df[treat_col] == 1
    control_mask = df[treat_col] == 0
    for cov in covariates:
        x_treated = df.loc[treated_mask, cov].astype(float)
        x_control = df.loc[control_mask, cov].astype(float)
        mean_t = x_treated.mean()
        if weights is None:
            mean_c = x_control.mean()
            var_t = x_treated.var(ddof=1)
            var_c = x_control.var(ddof=1)
        else:
            w = weights
            # Weights only apply to control observations; assign weight 1 to treated
            w_control = w[control_mask]
            mean_c = np.average(x_control, weights=w_control)
            # Compute weighted variance for control group
            # Weighted variance formula: sum(w*(x-mean)^2)/sum(w)
            var_c = np.sum(w_control * (x_control - mean_c) ** 2) / w_control.sum()
            # For treated group, use unweighted variance
            var_t = x_treated.var(ddof=1)
        pooled_std = np.sqrt((var_t + var_c) / 2.0)
        # Avoid division by zero
        smd[cov] = (mean_t - mean_c) / (pooled_std + 1e-8)
    return smd


def example_usage():
    """Demonstrate the use of the estimators on the NSW experimental data.

    This function is intended for interactive exploration.  It attempts
    to load the NSW treated and control samples from files named
    ``nswre74_treated.txt`` and ``nswre74_control.txt`` in the current
    working directory.  If the files are not found, the demonstration
    silently skips execution.

    When available, the function prints estimates from several methods
    discussed in the accompanying project description.  It also computes
    simple covariate balance diagnostics before and after weighting.
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
    # Unadjusted difference in means
    att_dm, mean_t, mean_c = difference_in_means(df)
    print(f"Difference in means estimate (ATT): {att_dm:.2f}")
    # OLS adjustment
    att_ols = ols_regression_att(df)
    print(f"OLS regression estimate: {att_ols:.2f}")
    # Propensity scores
    covariates = [
        "age",
        "educ",
        "black",
        "hispan",
        "married",
        "nodegree",
        "re74",
        "re75",
    ]
    ps = estimate_propensity_score(df, covariates=covariates)
    # Propensity score stratification
    att_strat = propensity_score_stratification_att(df, ps)
    print(f"Propensity score stratification estimate: {att_strat:.2f}")
    # Nearest neighbour matching with replacement
    att_match = propensity_score_matching_att(df, ps, replace=True)
    print(f"Nearest neighbour matching estimate: {att_match:.2f}")
    # Inverse probability weighting
    att_ipw = ipw_att(df, ps)
    print(f"IPW estimate: {att_ipw:.2f}")
    # Augmented inverse probability weighting
    att_aipw = aipw_att(df, ps, covariates=covariates)
    print(f"AIPW estimate: {att_aipw:.2f}")
    # Covariate balance diagnostics
    print("\nCovariate balance diagnostics (unweighted):")
    bal_unweighted = covariate_balance(df, covariates)
    for cov, smd_val in bal_unweighted.items():
        print(f"  {cov}: {smd_val:.3f}")
    # Weighted balance using IPW weights
    weights = np.where(df['treat'] == 1, 1.0, ps / (1.0 - ps + 1e-8))
    bal_weighted = covariate_balance(df, covariates, weights=weights)
    print("\nCovariate balance diagnostics (IPW weighted controls):")
    for cov, smd_val in bal_weighted.items():
        print(f"  {cov}: {smd_val:.3f}")


if __name__ == "__main__":
    # Run the demonstration when executed as a script
    example_usage()