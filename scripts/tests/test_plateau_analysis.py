import numpy as np
import pandas as pd

from plateau_analysis import sensitivity_table_to_latex


def make_sensitivity():
    """A table shaped like the output of sensitivity_table, for two conditions."""
    index = pd.MultiIndex.from_tuples(
        [("P1", "classic", "action"), ("P1", "odd one out", "greebles")],
        names=["observer", "task", "image_set"],
    )
    return pd.DataFrame(
        [[np.nan, np.nan, 900.0], [900.0, 450.0, 225.0]],
        index=index,
        columns=pd.Index([0.5, 1.0, 2.0], name="threshold_pp"),
    )


def test_sensitivity_latex_rows():
    latex = sensitivity_table_to_latex(make_sensitivity(), largest_sample=1800)
    assert "        P1 & action & $> 1800$ & $> 1800$ & 900 \\\\" in latex
    assert "        P1 & greebles & 900 & 450 & 225 \\\\" in latex


def test_sensitivity_latex_header_has_one_column_per_threshold():
    latex = sensitivity_table_to_latex(make_sensitivity(), largest_sample=1800)
    assert "\\begin{tabular}{llrrr}" in latex
    assert "Obs. & Image set & 0.5 pp & 1.0 pp & 2.0 pp \\\\" in latex
    assert "\\multicolumn{3}{c}{Plateau (trials) at threshold}" in latex


def test_sensitivity_latex_groups_rows_by_task_in_order():
    latex = sensitivity_table_to_latex(make_sensitivity(), largest_sample=1800)
    classic = latex.index("\\multicolumn{5}{l}{\\textit{classic}}")
    odd_one_out = latex.index("\\multicolumn{5}{l}{\\textit{odd one out}}")
    assert classic < latex.index("P1 & action") < odd_one_out < latex.index("P1 & greebles")


def test_sensitivity_latex_labels_and_reference():
    latex = sensitivity_table_to_latex(
        make_sensitivity(), largest_sample=1800,
        plateau_label="tab:main", label="tab:sensitivity",
    )
    assert "\\label{tab:sensitivity}" in latex
    assert "Supplementary Table~\\ref{tab:main}" in latex
    assert latex.startswith("\\begin{table}[htbp]")
    assert latex.endswith("\\end{table}")


def test_sensitivity_latex_skips_missing_task():
    sensitivity = make_sensitivity().iloc[[1]]
    latex = sensitivity_table_to_latex(sensitivity, largest_sample=1800)
    assert "textit{classic}" not in latex
    assert "textit{odd one out}" in latex
