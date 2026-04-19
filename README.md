# SDS 5320 Causal Inference Benchmark: LaLonde / NSW

This project benchmarks common observational estimators against the experimental benchmark from the National Supported Work (NSW) job-training study used by LaLonde and by Dehejia & Wahba. The goal is to estimate the average treatment effect on the treated (ATT) and compare how different adjustment strategies perform when treatment assignment is not randomized. The script `lalonde_analysis.py` provides a compact, course-level implementation of six estimators plus covariate balance diagnostics.

## Dataset (NSW / LaLonde, Dehejia–Wahba version)

The analysis is written for the Dehejia–Wahba sample conventions:

- NSW treated: **185** observations
- NSW experimental controls: **260** observations
- Optional non-experimental control groups (e.g., CPS or PSID) can be appended for observational benchmarking

### Required files for the built-in demo

Place these files in the project root (or pass paths directly to `load_lalonde_dw`):

- `nswre74_treated.txt`
- `nswre74_control.txt`

The loader expects whitespace-delimited columns in the standard order:

`treat age educ black hispan married nodegree re74 re75 re78`

### Where to obtain data

The files are distributed via NBER resources associated with the LaLonde / Dehejia–Wahba materials. In practice, direct scripted download may fail depending on server settings, so manual browser download is often simplest.

## Installation

Use a clean Python environment, then install dependencies:

```bash
pip install numpy pandas statsmodels scikit-learn
```

## Usage

You can run the script directly:

```bash
python lalonde_analysis.py
```

This triggers `example_usage()`, which attempts to read `nswre74_treated.txt` and `nswre74_control.txt` from the current directory.

Expected console output includes:

1. Treated/control outcome means (`re78`)
2. An ATT summary table for all six estimators
3. A covariate balance table with standardized mean differences (SMD), unweighted and IPW-weighted

If the text files are missing, the demo prints a short message and exits without error.

## Methods summary

| Estimator | Key identifying assumption (informal) | Function |
|---|---|---|
| Difference in means | Random assignment / no confounding | `difference_in_means` |
| OLS regression adjustment | Conditional mean model correctly specified | `ols_regression_att` |
| Propensity score stratification | Unconfoundedness + overlap + within-stratum comparability | `propensity_score_stratification_att` |
| Propensity score matching (NN) | Unconfoundedness + overlap + quality of matches | `propensity_score_matching_att` |
| IPW for ATT | Unconfoundedness + positivity + correct PS model | `ipw_att` |
| AIPW for ATT | Unconfoundedness + positivity + either PS model or outcome model correctly specified (double robustness) | `aipw_att` |

## File structure

- `lalonde_analysis.py` — data loader (NSW treated/control), ATT estimators, and balance diagnostics
- `README.md` — project overview, setup, and usage notes

## References

- LaLonde, R. J. (1986). *Evaluating the Econometric Evaluations of Training Programs with Experimental Data*. **American Economic Review**, 76(4), 604–620.
- Dehejia, R. H., & Wahba, S. (1999). *Causal Effects in Nonexperimental Studies: Reevaluating the Evaluation of Training Programs*. **Journal of the American Statistical Association**, 94(448), 1053–1062.
- Rosenbaum, P. R., & Rubin, D. B. (1983). *The Central Role of the Propensity Score in Observational Studies for Causal Effects*. **Biometrika**, 70(1), 41–55.
- Lunceford, J. K., & Davidian, M. (2004). *Stratification and Weighting via the Propensity Score in Estimation of Causal Treatment Effects: A Comparative Study*. **Statistics in Medicine**, 23(19), 2937–2960.
