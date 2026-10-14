"""Drawing helpers for the figures of the pooling notebooks.

Drawing only: the analysis decisions live in `size_controlled_pooling.py`
and the notebooks. Text uses the ink colours.
"""

import numpy as np

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e6e3'


def finish_axes(ax, ticks, tick_labels, xlabel=None, ylabel=None, title=None):
    """Recessive axes: no top or right spine, a light horizontal grid, ink text."""
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(min(ticks) - 0.5, max(ticks) + 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, loc='left', fontsize=10, color=INK)


# the colours of the image sets in the triplet-accuracy figures of the individual observers
PALETTE_IMAGE_SET = {'action': (0.4, 0.7607843137254902, 0.6470588235294118),
                     'greebles': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                     'taxonomic': (0.5529411764705883, 0.6274509803921569, 0.796078431372549)}


def draw_accuracy_panel(ax, test_scores, dimensions, significance_dimension, pvals_corrected, color, alpha=0.05):
    """Test accuracy against dimension, annotated as the figures of the individual observers.

    The line is the mean test accuracy over the cross-validation folds with
    error bars of 2 SD; the dashed vertical line marks the dimensionality of
    the significance criterion. Below the curve: the mean accuracy at every
    dimension and, for every step, the mean gain and the Holm-corrected
    p-value, both bold when the p-value is below `alpha`.
    """
    test_scores = np.asarray(test_scores)
    means = test_scores.mean(axis=1)
    ax.errorbar(dimensions, means, yerr=2 * test_scores.std(axis=1, ddof=1), color=color, lw=2, elinewidth=2)
    ax.axvline(significance_dimension, color=color, alpha=0.75, zorder=-1, ls='--')

    gains = np.diff(test_scores, axis=0).mean(axis=1)
    n = len(dimensions)
    for d in range(n):
        ax.annotate(f"{means[d] * 100:.1f} %", xy=(0.1 + 0.2 * d, 0.4), xycoords='axes fraction',
                    fontsize=8, ha='center', va='center')
    for d in range(n - 1):
        weight = 'bold' if pvals_corrected[d] < alpha else 'normal'
        x = 0.15 + 0.25 * d
        ax.annotate(f"{gains[d] * 100:.1f} %", xy=(x, 0.3), xycoords='axes fraction',
                    fontsize=8, weight=weight, ha='center', va='center')
        ax.annotate(f"{pvals_corrected[d]:.3f}", xy=(x, 0.2), xycoords='axes fraction',
                    fontsize=8, weight=weight, ha='center', va='center')
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(0.75, n + 0.5)
    ax.set_xticks(dimensions)
