"""Plateau analysis for the Experiment 2 sub-sampling data.

This module locates, for each observer, task and image set, the number of
trials at which triplet accuracy stops improving appreciably, and expresses
that number relative to the d * N * log(N) sample complexity bound.

Plateau criterion
-----------------
The sub-sample sizes form a geometric grid (28, 56, 112, 225, 450, 900,
1800), so every step is approximately a doubling of the data. We therefore
express the improvement of a step as a gain per doubling,

    gain[i] = (accuracy[i + 1] - accuracy[i]) / log2(n[i + 1] / n[i])

and define the plateau as the smallest sample size from which every
subsequent doubling gains less than a threshold Delta. Requiring the
condition to hold for all later steps, rather than only for the next one,
prevents a single noisy dip from declaring an early plateau.

Why the full data set point is excluded
---------------------------------------
The reference embedding fitted on the complete set of triplets is evaluated
in-sample, whereas every sub-sampled point is evaluated on held-out triplets.
The two accuracies are therefore not on the same scale, and the apparent gain
from the largest sub-sample to the complete set is inflated by that
difference rather than by additional data. In the empirical data this
inflation is large (up to 5.5 percentage points per doubling in the classic
conditions, exceeding the two preceding steps). The complete data set point
is consequently used only as a ceiling marker and never enters the gain
computation. Do not add it back.

Units
-----
Functions that take accuracies are unit agnostic: gains and thresholds are
returned and interpreted in whatever unit the accuracies are supplied in.
The table builders in this module work in percentage points.

The d * N * log(N) bound counts triplet comparisons. One classic trial yields
one triplet, one odd-one-out trial yields two, so the bound expressed in
trials is halved for the odd-one-out task. Ratios of plateau to bound are
unaffected as long as both are expressed in the same unit.

Scope
-----
This module computes and tabulates; it never draws. It therefore imports no
plotting library, and the numbers it produces cannot be changed by a change to
a figure. The drawing helpers for the sub-sampling figures live in
`subsampling_plots`.
"""

import glob
import os

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD_PP = 1.0
DEFAULT_N_STIMULI = 18
DEFAULT_RUN_SIZES = (10, 50, 40)
TRIPLETS_PER_TRIAL = {"classic": 1, "odd one out": 2}


def _validated_curve(sample_sizes, accuracies):
    """Return sample sizes and accuracies as float arrays after checking them."""
    sizes = np.asarray(sample_sizes, dtype=float)
    values = np.asarray(accuracies, dtype=float)

    if sizes.ndim != 1 or values.ndim != 1:
        raise ValueError("sample_sizes and accuracies must be one dimensional")
    if sizes.size != values.size:
        raise ValueError(
            "sample_sizes and accuracies must have the same length, got "
            "{0} and {1}".format(sizes.size, values.size)
        )
    if sizes.size < 2:
        raise ValueError("at least two sample sizes are required")
    if np.any(sizes <= 0):
        raise ValueError("sample sizes must be strictly positive")
    if np.any(np.diff(sizes) <= 0):
        raise ValueError("sample_sizes must be strictly increasing")

    return sizes, values


def gains_per_doubling(sample_sizes, accuracies):
    """Accuracy gain per doubling of the sample size, for each step.

    Parameters
    ----------
    sample_sizes : sequence of float
        Strictly increasing sample sizes.
    accuracies : sequence of float
        Accuracy at each sample size, in any unit.

    Returns
    -------
    numpy.ndarray
        Array of length len(sample_sizes) - 1. Element i is the accuracy
        change from sample_sizes[i] to sample_sizes[i + 1], divided by the
        number of doublings that step corresponds to.
    """
    sizes, values = _validated_curve(sample_sizes, accuracies)
    doublings = np.log2(sizes[1:] / sizes[:-1])
    return np.diff(values) / doublings


def plateau_onset(sample_sizes, accuracies, threshold):
    """Smallest sample size from which all later doublings gain less than threshold.

    Parameters
    ----------
    sample_sizes : sequence of float
        Strictly increasing sample sizes.
    accuracies : sequence of float
        Accuracy at each sample size, in the same unit as threshold.
    threshold : float
        Maximum gain per doubling that still counts as a plateau.

    Returns
    -------
    float or None
        The sample size at which the plateau starts, or None if no such
        sample size exists within the supplied range.
    """
    sizes, values = _validated_curve(sample_sizes, accuracies)
    gains = gains_per_doubling(sizes, values)
    below = gains < threshold

    onset_index = None
    for index in range(below.size - 1, -1, -1):
        if below[index]:
            onset_index = index
        else:
            break

    if onset_index is None:
        return None
    return float(sizes[onset_index])


def sample_complexity_bound_triplets(n_stimuli, dimensionality):
    """The d * N * log(N) bound, in triplet comparisons.

    Uses the natural logarithm and truncates towards zero, matching the
    convention used elsewhere in the analysis code.
    """
    if n_stimuli < 2:
        raise ValueError("n_stimuli must be at least 2")
    if dimensionality < 1:
        raise ValueError("dimensionality must be at least 1")
    return int(dimensionality * n_stimuli * np.log(n_stimuli))


def triplets_to_trials(n_triplets, task):
    """Convert a number of triplet comparisons into a number of trials.

    One classic trial yields one triplet, one odd-one-out trial yields two.
    """
    if task not in TRIPLETS_PER_TRIAL:
        raise KeyError(
            "unknown task {0!r}, expected one of {1}".format(
                task, sorted(TRIPLETS_PER_TRIAL)
            )
        )
    return n_triplets / TRIPLETS_PER_TRIAL[task]


def load_subsampling_runs(data_dir, run_sizes=DEFAULT_RUN_SIZES):
    """Read and concatenate the per condition sub-sampling run files."""
    pattern = os.path.join(data_dir, "subsampling_analysis-*.csv")
    paths = sorted(
        path
        for path in glob.glob(pattern)
        if not path.endswith("triplet_accuracy_alldata.csv")
    )
    if not paths:
        raise FileNotFoundError("no sub-sampling run files found in {0}".format(data_dir))

    expected_suffixes = tuple("-{0}.csv".format(size) for size in run_sizes)
    paths = [path for path in paths if path.endswith(expected_suffixes)]
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def load_reference_embeddings(data_dir):
    """Read the reference embeddings fitted on the complete triplet set."""
    path = os.path.join(data_dir, "subsampling_analysis-triplet_accuracy_alldata.csv")
    return pd.read_csv(path)


def reference_row(df_reference, observer, task, image_set):
    """The complete data set row for one observer, task and image set.

    The figures need the dimensionality, the ceiling accuracy and the total
    number of trials of a condition, all of which live in this one row.
    Raises rather than returning the first match silently, because a duplicated
    condition would mean the reference table was built twice.
    """
    selection = df_reference[
        (df_reference["observer"] == observer)
        & (df_reference["task"] == task)
        & (df_reference["image_set"] == image_set)
    ]
    if len(selection) != 1:
        raise ValueError(
            "expected exactly one reference row for observer={0!r}, task={1!r}, "
            "image_set={2!r}, got {3}".format(observer, task, image_set, len(selection))
        )
    return selection.iloc[0]


def mean_accuracy_curve(df_runs, observer, task, image_set, accuracy_scale=100.0):
    """Mean accuracy across random runs, per sample size, for one condition.

    Returns
    -------
    tuple of numpy.ndarray
        Sample sizes in ascending order, and the corresponding mean accuracy
        multiplied by accuracy_scale (percentage points by default).
    """
    selection = df_runs[
        (df_runs["observer"] == observer)
        & (df_runs["task"] == task)
        & (df_runs["image_set"] == image_set)
    ]
    if selection.empty:
        raise ValueError(
            "no rows for observer={0!r}, task={1!r}, image_set={2!r}".format(
                observer, task, image_set
            )
        )
    curve = selection.groupby("n")["accuracy"].mean().sort_index()
    return curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float) * accuracy_scale


def condition_summary(
    df_runs,
    reference_row,
    threshold_pp=DEFAULT_THRESHOLD_PP,
    n_stimuli=DEFAULT_N_STIMULI,
):
    """Summarise the plateau for a single observer, task and image set."""
    observer = reference_row["observer"]
    task = reference_row["task"]
    image_set = reference_row["image_set"]
    dimensionality = int(reference_row["d"])

    sizes, accuracies = mean_accuracy_curve(df_runs, observer, task, image_set)
    onset = plateau_onset(sizes, accuracies, threshold_pp)

    bound_triplets = sample_complexity_bound_triplets(n_stimuli, dimensionality)
    bound_trials = triplets_to_trials(bound_triplets, task)

    if onset is None:
        accuracy_at_plateau = np.nan
        ratio = np.nan
    else:
        accuracy_at_plateau = float(accuracies[sizes == onset][0])
        ratio = onset / bound_trials

    return {
        "observer": observer,
        "task": task,
        "image_set": image_set,
        "d": dimensionality,
        "bound_triplets": bound_triplets,
        "bound_trials": bound_trials,
        "plateau_trials": onset,
        "plateau_reached": onset is not None,
        "ratio_to_bound": ratio,
        "largest_sample": float(sizes[-1]),
        "lower_bound_ratio": float(sizes[-1]) / bound_trials,
        "accuracy_at_plateau": accuracy_at_plateau,
        "ceiling_accuracy": float(reference_row["accuracy"]) * 100.0,
    }


def plateau_table(
    df_runs,
    df_reference,
    threshold_pp=DEFAULT_THRESHOLD_PP,
    n_stimuli=DEFAULT_N_STIMULI,
):
    """Plateau summary for every condition present in the reference table."""
    records = [
        condition_summary(df_runs, row, threshold_pp, n_stimuli)
        for _, row in df_reference.iterrows()
    ]
    return pd.DataFrame.from_records(records)


def sensitivity_table(
    df_runs,
    df_reference,
    thresholds_pp,
    n_stimuli=DEFAULT_N_STIMULI,
):
    """Plateau sample size per condition for a range of thresholds."""
    frames = []
    for threshold in thresholds_pp:
        table = plateau_table(df_runs, df_reference, threshold, n_stimuli)
        table["threshold_pp"] = threshold
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    return combined.pivot_table(
        index=["observer", "task", "image_set"],
        columns="threshold_pp",
        values="plateau_trials",
        dropna=False,
    )


def bound_interval_trials(n_stimuli, dimensionality, task, upper_factor=2.0):
    """The sample complexity interval of Haghiri et al., expressed in trials.

    Returns the pair (d * N * log(N), upper_factor * d * N * log(N)) after
    conversion from triplet comparisons into trials, which halves both
    endpoints for the odd-one-out task. Use this rather than the raw triplet
    bound whenever the quantity is drawn on an axis that counts trials.
    """
    bound = sample_complexity_bound_triplets(n_stimuli, dimensionality)
    lower = triplets_to_trials(bound, task)
    return lower, upper_factor * lower


def plateau_lookup(table):
    """Map (observer, task, image_set) to the plateau sample size.

    Conditions that never reached the criterion map to None.
    """
    lookup = {}
    for _, row in table.iterrows():
        key = (row["observer"], row["task"], row["image_set"])
        if row["plateau_reached"]:
            lookup[key] = float(row["plateau_trials"])
        else:
            lookup[key] = None
    return lookup


def _do_not_plateau_label(row, unreached_marker):
    largest = int(row["largest_sample"])
    return unreached_marker.format(largest=largest, ratio=row["lower_bound_ratio"])


def plateau_table_to_latex(
    table,
    threshold_pp=DEFAULT_THRESHOLD_PP,
    unreached_marker="$> {largest}$",
    label="tab:subsampling_plateau",
):
    """Render the plateau table as a LaTeX tabular block."""
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\begin{tabular}{llrrrrr}",
        "        \\hline",
        "        Obs. & Image set & $d$ & Bound & Bound & Plateau & Plateau \\\\",
        "         &  &  & (triplets) & (trials) & (trials) & / bound \\\\",
        "        \\hline",
    ]

    for task in ["classic", "odd one out"]:
        subset = table[table["task"] == task]
        if subset.empty:
            continue
        lines.append(
            "        \\multicolumn{{7}}{{l}}{{\\textit{{{0}}}}} \\\\".format(task)
        )
        for _, row in subset.iterrows():
            if row["plateau_reached"]:
                plateau_text = "{0:.0f}".format(row["plateau_trials"])
                ratio_text = "{0:.1f}".format(row["ratio_to_bound"])
            else:
                plateau_text = _do_not_plateau_label(row, unreached_marker)
                ratio_text = "$> {0:.1f}$".format(row["lower_bound_ratio"])
            lines.append(
                "        {0} & {1} & {2} & {3} & {4:.0f} & {5} & {6} \\\\".format(
                    row["observer"],
                    row["image_set"],
                    row["d"],
                    row["bound_triplets"],
                    row["bound_trials"],
                    plateau_text,
                    ratio_text,
                )
            )
        lines.append("        \\hline")

    caption = (
        "Sample size at which triplet accuracy plateaus, per observer, task and "
        "image set. The plateau is defined as the smallest sub-sample size from "
        "which every further doubling of the data increases mean triplet accuracy "
        "by less than {0:.1f} percentage points. Gains are computed on the "
        "sub-sample grid only; the complete data set point is evaluated in-sample "
        "and is therefore not comparable to the held-out accuracies. Bound denotes "
        "$d\\,N\\log(N)$ with $N = 18$ and the dimensionality of the reference "
        "embedding, given both in triplets and in trials, since one odd-one-out "
        "trial yields two triplets. Entries marked with $>$ did not reach the "
        "criterion within the sampled range."
    ).format(threshold_pp)

    lines.extend(
        [
            "    \\end{tabular}",
            "    \\caption{{{0}}}".format(caption),
            "    \\label{{{0}}}".format(label),
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def sensitivity_table_to_latex(
    sensitivity,
    largest_sample,
    plateau_label="tab:subsampling_plateau",
    label="tab:subsampling_plateau_sensitivity",
):
    """Render the output of sensitivity_table as a LaTeX tabular block.

    One column per threshold, in the order of the columns of sensitivity, with
    the plateau in trials. Conditions that did not reach the criterion within
    the sampled range (NaN) are written as "> largest_sample". Rows are grouped
    by task, as in plateau_table_to_latex.
    """
    thresholds = list(sensitivity.columns)
    n_columns = 2 + len(thresholds)
    unreached = "$> {0:d}$".format(int(largest_sample))
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\begin{tabular}{ll" + "r" * len(thresholds) + "}",
        "        \\hline",
        "         &  & \\multicolumn{{{0}}}{{c}}{{Plateau (trials) at threshold}} \\\\".format(
            len(thresholds)
        ),
        "        Obs. & Image set & "
        + " & ".join("{0:.1f} pp".format(threshold) for threshold in thresholds)
        + " \\\\",
        "        \\hline",
    ]

    tasks = sensitivity.index.get_level_values("task")
    for task in ["classic", "odd one out"]:
        subset = sensitivity[tasks == task]
        if subset.empty:
            continue
        lines.append(
            "        \\multicolumn{{{0}}}{{l}}{{\\textit{{{1}}}}} \\\\".format(n_columns, task)
        )
        for (observer, _, image_set), row in subset.iterrows():
            cells = [
                unreached if np.isnan(row[threshold]) else "{0:.0f}".format(row[threshold])
                for threshold in thresholds
            ]
            lines.append(
                "        {0} & {1} & {2} \\\\".format(observer, image_set, " & ".join(cells))
            )
        lines.append("        \\hline")

    caption = (
        "Sample size at which triplet accuracy plateaus for different values of the "
        "threshold, per observer, task and image set. The plateau is defined as in "
        "Supplementary Table~\\ref{{{0}}}, as the smallest sub-sample size from which "
        "every further doubling of the data increases mean triplet accuracy by less "
        "than the threshold, given in percentage points (pp). Entries marked with $>$ "
        "did not reach the criterion within the sampled range."
    ).format(plateau_label)

    lines.extend(
        [
            "    \\end{tabular}",
            "    \\caption{{{0}}}".format(caption),
            "    \\label{{{0}}}".format(label),
            "\\end{table}",
        ]
    )
    return "\n".join(lines)
