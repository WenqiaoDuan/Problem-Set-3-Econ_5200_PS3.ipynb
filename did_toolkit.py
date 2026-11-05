"""Difference-in-differences toolkit, extending Lab 11."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

NEVER = 10**9


# Lab 11 functions
def two_by_two(df: pd.DataFrame, outcome: str,
               treated: str, post: str) -> float:
    """Calculate the canonical 2x2 DiD estimate."""
    g = df.groupby([treated, post])[outcome].mean()
    return float((g[1, 1] - g[1, 0]) - (g[0, 1] - g[0, 0]))


def att_by_cohort_period(df: pd.DataFrame, time: str,
                         outcome: str, cohort: str) -> pd.DataFrame:
    """Estimate ATT(g,t) using only not-yet-treated controls."""
    out = []
    cohorts = sorted(c for c in df[cohort].unique() if c < NEVER)

    for g in cohorts:
        tr = df[df[cohort] == g]

        for t in sorted(df.loc[df[time] >= g, time].unique()):
            co = df[df[cohort] > t]

            if co.empty or (tr[time] == g - 1).sum() == 0:
                continue

            d_tr = (
                tr.loc[tr[time] == t, outcome].mean()
                - tr.loc[tr[time] == g - 1, outcome].mean()
            )
            d_co = (
                co.loc[co[time] == t, outcome].mean()
                - co.loc[co[time] == g - 1, outcome].mean()
            )

            out.append({
                "g": g, "t": t, "e": t - g,
                "att": d_tr - d_co,
                "n": int((tr[time] == t).sum())
            })

    return pd.DataFrame(out)


def overall_att(cells: pd.DataFrame) -> float:
    """Calculate the size-weighted average ATT."""
    return float(np.average(cells["att"], weights=cells["n"]))


def event_study(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ATT estimates by event time."""
    rows = []
    for e, group in cells.groupby("e"):
        rows.append({
            "e": e,
            "att": np.average(group["att"], weights=group["n"])
        })
    return pd.DataFrame(rows)


def cluster_se_ratio(fit_iid, fit_clustered, param: str) -> float:
    """Calculate the ratio of clustered to default standard errors."""
    return float(fit_clustered.bse[param] / fit_iid.bse[param])


# Notebook function
def clean_att(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Estimate clean ATT using not-yet-treated controls."""
    cells = att_by_cohort_period(df, "t", "y", "g")
    return cells, overall_att(cells)


# Step 3.3
def clustered_inference(
    df: pd.DataFrame,
    formula: str,
    cluster_col: str
) -> RegressionResultsWrapper:
    """Fit OLS with cluster-robust standard errors."""
    if cluster_col not in df.columns:
        raise ValueError(f"Column {cluster_col} not found")

    return smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]}
    )


if __name__ == "__main__":
    test = pd.DataFrame({
        "treated": [0, 0, 1, 1],
        "post": [0, 1, 0, 1],
        "y": [10, 12, 11, 15]
    })

    assert np.isclose(
        two_by_two(test, "y", "treated", "post"), 2.0
    )

    test_cells = pd.DataFrame({
        "e": [0, 1],
        "att": [1.0, 2.0],
        "n": [10, 10]
    })

    es = event_study(test_cells)
    assert len(es) == 2
    assert np.isclose(es["att"].mean(), 1.5)

    print("PASS: Basic toolkit assertions")


import itertools
import statsmodels.api as sm

def wild_cluster_bootstrap(
    panel,
    outcome,
    treatment,
    cluster,
    controls=None,
    alpha=0.05
):
    """
    Exact null-imposed wild cluster bootstrap.

    Uses:
      - Rademacher weights at the cluster level
      - All 2^G possible weight vectors
      - Restricted-model residuals
      - Cluster-robust t-statistics

    For 8 clusters: 2^8 = 256 bootstrap replications.
    """

    controls = [] if controls is None else list(controls)

    columns = list(dict.fromkeys(
        [outcome, treatment, cluster] + controls
    ))

    df = panel[columns].dropna().copy()

    groups, cluster_ids = pd.factorize(df[cluster])
    G = len(cluster_ids)

    if G < 2:
        raise ValueError("At least two clusters are required.")

    if G > 12:
        raise ValueError(
            "Exact enumeration is limited to 12 clusters."
        )

    y = df[outcome].to_numpy(dtype=float)

    X = sm.add_constant(
        df[[treatment] + controls].astype(float),
        has_constant="add"
    )

    X = X.loc[:, ~X.columns.duplicated()]

    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("Design matrix is rank deficient.")

    # 1. Unrestricted model and observed cluster-robust t
    unrestricted = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True
        },
        use_t=True
    )

    beta_hat = float(unrestricted.params[treatment])
    observed_t = float(unrestricted.tvalues[treatment])
    naive_p = float(unrestricted.pvalues[treatment])

    # 2. Impose H0: beta_treatment = 0
    restricted_columns = [
        c for c in X.columns if c != treatment
    ]

    X_restricted = X[restricted_columns]

    restricted = sm.OLS(y, X_restricted).fit()

    y_hat_null = np.asarray(restricted.fittedvalues)
    residuals_null = np.asarray(restricted.resid)

    # 3. Enumerate every Rademacher weight vector
    weight_vectors = list(
        itertools.product([-1.0, 1.0], repeat=G)
    )

    bootstrap_t = []

    # 4. Generate bootstrap samples under the null
    for weights in weight_vectors:

        weights = np.asarray(weights)

        # One weight per cluster, not per observation
        observation_weights = weights[groups]

        y_star = (
            y_hat_null
            + residuals_null * observation_weights
        )

        # 5. Re-estimate unrestricted model
        boot_model = sm.OLS(y_star, X).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": groups,
                "use_correction": True,
                "df_correction": True
            },
            use_t=True
        )

        t_star = float(
            boot_model.params[treatment]
            / boot_model.bse[treatment]
        )

        bootstrap_t.append(t_star)

    bootstrap_t = np.asarray(bootstrap_t)

    # 6. Exact two-sided bootstrap p-value
    bootstrap_p = float(
        np.mean(
            np.abs(bootstrap_t) >= abs(observed_t) - 1e-12
        )
    )

    naive_reject = naive_p < alpha
    bootstrap_reject = bootstrap_p < alpha

    results = {
        "coefficient": beta_hat,
        "observed_t": observed_t,
        "naive_cluster_p": naive_p,
        "wild_cluster_bootstrap_p": bootstrap_p,
        "n_clusters": G,
        "n_bootstrap": len(weight_vectors),
        "alpha": alpha,
        "naive_significant": naive_reject,
        "bootstrap_significant": bootstrap_reject,
        "conclusion_changed": (
            naive_reject != bootstrap_reject
        )
    }

    print("\n=== Wild Cluster Bootstrap Results ===")
    print(f"Treatment coefficient: {beta_hat:.6f}")
    print(f"Observed t-statistic:  {observed_t:.4f}")
    print(f"Number of clusters:    {G}")
    print(f"Bootstrap replications: {len(weight_vectors)}")
    print(f"Naive cluster p-value: {naive_p:.6f}")
    print(f"Wild bootstrap p-value: {bootstrap_p:.6f}")

    print("\n=== Inference Comparison ===")

    print(
        "Naive cluster-robust:",
        "Significant" if naive_reject else "Not significant"
    )

    print(
        "Wild cluster bootstrap:",
        "Significant" if bootstrap_reject else "Not significant"
    )

    if naive_reject == bootstrap_reject:
        print("Conclusion: The finding is unchanged.")
    elif naive_reject and not bootstrap_reject:
        print(
            "Conclusion: The original significant finding "
            "does not survive corrected inference."
        )
    else:
        print(
            "Conclusion: The finding becomes significant "
            "under wild cluster bootstrap inference."
        )

    return results, bootstrap_t

