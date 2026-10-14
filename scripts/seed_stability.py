"""How stable is the estimated dimensionality against the randomness in the fit?

A fit contains two random ingredients, both fixed by the seeds passed to
`dimensionality_estimation.estimate_dimensionality`:

- the initial embeddings SOE starts its optimisation from (`soe_random_state`),
- the cross-validation splits (`random_state`).

Neither is a property of the data, so if the estimate moves when they change,
the estimate is not determined by the data alone. This module refits a
condition several times, varying one ingredient at a time, and reports the
distribution of the estimated dimensionality.

For conditions whose estimate depends on the SOE initialisation, the analysis
reports the majority dimensionality over SOE seeds 0-9 with the
cross-validation splits at seed 0, fitted with the lowest SOE seed that gives
that majority (`majority_rule`). Run this module from `scripts/` to refit the
conditions in `UNSTABLE_CONDITIONS` on the published inputs; fits already in
`RESULTS_FILE` are skipped, so an interrupted run can be resumed.
"""

import ast
import os

import numpy as np
import pandas as pd

from dimensionality_estimation import check_seed, estimate_dimensionality

VARY = ("both", "soe", "cv")
SEEDS = range(10)
UNSTABLE_CONDITIONS = [("O2", "odd one out", "greebles"),
                       ("O4", "odd one out", "taxonomic"),
                       ("P1", "odd one out", "action")]
MASTER_TABLE = "../data/found-dimensionality-embeddings-aggregated-per-observer-nsplit-5-nrepeat-30.csv"
RESULTS_FILE = "../data/dimensionality_seed_stability_published_inputs.csv"
KEYS = ["observer", "task", "image_set"]


def seeds_for(vary, seed, base=0):
    """(soe_seed, cv_seed) for one run of a given variation mode."""
    seed, base = check_seed(seed), check_seed(base)
    if vary == "both":
        return seed, seed
    if vary == "soe":
        return seed, base
    if vary == "cv":
        return base, seed
    raise ValueError("unknown mode {0!r}, expected one of {1}".format(vary, VARY))


def fit_once(triplets, soe_seed, cv_seed, **fit_kwargs):
    """One cross-validated fit with the two random ingredients seeded separately.

    Returns the cblearn result object of `estimate_dimensionality`.
    """
    dim_result, _ = estimate_dimensionality(triplets, random_state=cv_seed,
                                            soe_random_state=soe_seed, **fit_kwargs)
    return dim_result


def stability_runs(triplets, seeds, vary, label=None, **fit_kwargs):
    """Refit one condition once per seed, varying `vary`.

    Returns
    -------
    pandas.DataFrame
        One row per run with the seeds used, the estimated dimensionality and
        the Holm-corrected p-value of every sequential test.
    """
    records = []
    for seed in seeds:
        soe_seed, cv_seed = seeds_for(vary, seed)
        result = fit_once(triplets, soe_seed, cv_seed, **fit_kwargs)
        record = {"vary": vary, "seed": int(seed), "soe_seed": soe_seed, "cv_seed": cv_seed,
                  "dimensionality": int(result.estimated_dimension)}
        if label is not None:
            record.update(label)
        for step, pval in enumerate(np.asarray(result.stats_result["pvals_corrected"], dtype=float)):
            record["p_{0}to{1}".format(step + 1, step + 2)] = float(pval)
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarise(runs):
    """Per condition and variation mode: the range of estimates over the seeds."""
    keys = [k for k in KEYS + ["vary"] if k in runs.columns]
    grouped = runs.groupby(keys)["dimensionality"]
    table = grouped.agg(n_runs="count", dim_min="min", dim_max="max",
                        dim_mode=lambda values: int(values.mode().min()))
    table["n_distinct"] = grouped.nunique()
    table["unstable"] = table["dim_min"] != table["dim_max"]
    return table.reset_index()


def majority_rule(runs, cv_seed=0):
    """The dimensionality to report for each condition, from the SOE-seed runs.

    Uses the runs that vary the SOE seed with the cross-validation splits at
    `cv_seed`. Per condition it returns the majority dimensionality (ties go
    to the lower one), the share of seeds that give it, and the lowest SOE
    seed that gives it, which is the seed the condition is fitted with.
    """
    selection = runs[(runs["vary"] == "soe") & (runs["cv_seed"] == cv_seed)]
    if selection.empty:
        raise ValueError("no runs varying the SOE seed with cv_seed={0}".format(cv_seed))
    records = []
    for key, group in selection.groupby(KEYS, sort=False):
        counts = group["dimensionality"].value_counts()
        majority = int(counts[counts == counts.max()].index.min())
        share = float(counts[majority] / len(group))
        soe_seed = int(group.loc[group["dimensionality"] == majority, "soe_seed"].min())
        records.append(dict(zip(KEYS, key), n_seeds=len(group), dimensionality=majority,
                            share=share, soe_seed=soe_seed))
    return pd.DataFrame.from_records(records)


def run_unstable_conditions(master_table=MASTER_TABLE, results_file=RESULTS_FILE,
                            conditions=UNSTABLE_CONDITIONS, seeds=SEEDS, modes=("soe", "cv")):
    """Refit the given conditions, appending one row per fit to results_file.

    The triplets are the published ones, read from the master table. Fits
    already in results_file are skipped.
    """
    table = pd.read_csv(master_table).set_index(KEYS)
    done = set()
    if os.path.exists(results_file):
        previous = pd.read_csv(results_file)
        done = set(zip(*(previous[k] for k in KEYS + ["vary", "seed"])))
    for key in conditions:
        triplets = np.array(ast.literal_eval(table.loc[key, "triplets"])).astype(int)
        for vary in modes:
            for seed in seeds:
                if (*key, vary, seed) in done:
                    continue
                row = stability_runs(triplets, [seed], vary, label=dict(zip(KEYS, key)))
                row.to_csv(results_file, mode="a", header=not os.path.exists(results_file),
                           index=False)
                print(" / ".join(key), vary, seed, "->", int(row["dimensionality"].iloc[0]),
                      flush=True)


if __name__ == "__main__":
    run_unstable_conditions()
    runs = pd.read_csv(RESULTS_FILE)
    print(summarise(runs).to_string(index=False))
    print(majority_rule(runs).to_string(index=False))
