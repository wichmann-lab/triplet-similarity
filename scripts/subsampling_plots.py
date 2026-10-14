"""Figure helpers for the Experiment 2 sub-sampling figures.

Everything in this module draws; nothing in it decides anything about the
analysis. The counterpart module `plateau_analysis` computes the plateau, the
sample complexity bound and the summary tables, and imports no plotting
library. Keeping the two apart means the analysis can be tested and re-run
without a display, and that changing the look of a figure cannot silently
change a published number.

The division of labour with the notebook is deliberate as well. This module
owns the drawing mechanics that would otherwise be repeated once per panel:
the dash pattern of a plateau line, the three calls that make one bound
interval, the proxy handles a manually placed legend needs. The notebook keeps
the layout decisions that differ from figure to figure, such as where along
the ordinate an annotation sits and which panels exist at all.
"""

from matplotlib.lines import Line2D

# The three image sets, in the order they appear in the figures, with the
# first three colours of the seaborn Set2 palette written out. They are
# written out rather than looked up so that a change of seaborn default
# cannot alter the colours of already published figures.
PALETTE_IMAGE_SET = {
    "action": (0.4, 0.7607843137254902, 0.6470588235294118),
    "greebles": (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
    "taxonomic": (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
}

# Distinct dash patterns per image set, so that two conditions that plateau at
# the same sample size remain distinguishable where their lines coincide.
PLATEAU_DASHES = {
    "action": (5, 2),
    "greebles": (2, 2),
    "taxonomic": (1, 2),
}

OBS_TASK_SEPARATOR = " - "


def obs_task_label(observer, task):
    """The combined facet label used by the single row figures."""
    return "{0}{1}{2}".format(observer, OBS_TASK_SEPARATOR, task)


def add_obs_task_column(df, column="obs-task"):
    """Add the combined observer and task column used as a facet variable.

    Returns the same DataFrame, modified in place, so it can be used inline.
    """
    df[column] = [
        obs_task_label(observer, task)
        for observer, task in zip(df["observer"], df["task"])
    ]
    return df


def split_obs_task(label):
    """Recover (observer, task) from a combined facet label.

    Splitting on the first separator only, because the task name itself
    contains spaces ("odd one out").
    """
    observer, separator, task = label.partition(OBS_TASK_SEPARATOR)
    if not separator:
        raise ValueError(
            "{0!r} is not an observer and task label separated by {1!r}".format(
                label, OBS_TASK_SEPARATOR
            )
        )
    return observer, task


def plateau_line_style(color, image_set=None, **overrides):
    """Line properties of a plateau line, as a keyword dictionary.

    Separated from the drawing so that the choice of dash pattern can be
    checked without inspecting a matplotlib artist, which exposes no public
    getter for its dash sequence.
    """
    settings = {"color": color, "linewidth": 1.2, "alpha": 0.9, "zorder": 1}
    if image_set in PLATEAU_DASHES:
        settings["dashes"] = PLATEAU_DASHES[image_set]
    else:
        settings["linestyle"] = "--"
    settings.update(overrides)
    return settings


def draw_plateau_line(axis, plateau_trials, color, image_set=None, **kwargs):
    """Draw the plateau as a vertical line on one axis.

    Does nothing and returns None if plateau_trials is None, which is how
    conditions that never reached the criterion are represented.
    """
    if plateau_trials is None:
        return None

    return axis.axvline(plateau_trials, **plateau_line_style(color, image_set, **kwargs))


def draw_bound_marker(axis, bound_trials, y, color, marker="*", markersize=5, **kwargs):
    """Mark the sample complexity bound with a single symbol at height y.

    Used where only the lower end of the bound is shown. The abscissa counts
    trials, so pass the bound already converted into trials.
    """
    settings = {"markersize": markersize, "zorder": 2}
    settings.update(kwargs)
    return axis.plot([bound_trials], [y], marker, color=color, **settings)


def draw_bound_interval(axis, interval, y, color, markersize=4, **kwargs):
    """Draw the sample complexity interval as a bar with arrow heads at height y.

    interval is the (lower, upper) pair in trials, as returned by
    plateau_analysis.bound_interval_trials. Returns the three artists, so that
    a caller can restyle or remove them.
    """
    lower, upper = interval
    settings = {"zorder": 2}
    settings.update(kwargs)
    left = axis.plot([lower], [y], "<", color=color, markersize=markersize, **settings)
    right = axis.plot([upper], [y], ">", color=color, markersize=markersize, **settings)
    bar = axis.plot(
        [lower, upper], [y, y], color=color, markersize=0, **settings
    )
    return left, right, bar


def draw_ceiling_point(axis, n_trials, accuracy, color, markersize=7, **kwargs):
    """Mark the complete data set with a filled circle.

    The complete data set embedding is evaluated in-sample, so it is a ceiling
    marker rather than another point on the sub-sampling curve. See the
    plateau_analysis module docstring.
    """
    settings = {"markersize": markersize, "zorder": 2}
    settings.update(kwargs)
    return axis.plot([n_trials], [accuracy], "o", color=color, **settings)


def draw_ceiling_line(axis, accuracy, color, xmin, xmax, alpha=0.5, **kwargs):
    """Mark the complete data set accuracy as a horizontal line across the panel."""
    settings = {"alpha": alpha, "zorder": -1}
    settings.update(kwargs)
    return axis.hlines(y=accuracy, xmin=xmin, xmax=xmax, color=color, **settings)


def image_set_legend_handles(palette=None, **line_kwargs):
    """Proxy line handles, one per image set, for a manually placed legend.

    The handles mimic the appearance of the accuracy curves, so that a legend
    built from them matches the plotted data without relying on the automatic
    legend of the underlying grid.
    """
    if palette is None:
        palette = PALETTE_IMAGE_SET

    settings = {"marker": "o", "linewidth": 2, "markersize": 6}
    settings.update(line_kwargs)
    return [
        Line2D([0], [0], color=color, label=image_set, **settings)
        for image_set, color in palette.items()
    ]


def _clear_grid_legend(grid):
    """Remove a legend seaborn added, so a manual one can replace it."""
    if grid.legend is not None:
        grid.legend.remove()
        grid._legend = None


def place_legend_below(
    grid, palette=None, title="Image set", ncol=None, x=0.5, y=-0.14, **kwargs
):
    """Put the legend underneath the panels rather than beside them.

    A legend placed to the right of a single row of panels consumes figure
    width, which is the scarce dimension in a journal column. Laying the
    entries out horizontally underneath costs a little height instead.

    Any legend created previously by the grid is removed first, so this can be
    called in place of grid.add_legend. The default offset y clears a figure
    level x label placed at y = 0.06; move both together if either changes.

    x is the figure coordinate the legend is centred on, 0 being the left edge
    and 1 the right. It defaults to the middle, directly under a centred figure
    level x label; move it aside when the two would otherwise sit on top of
    each other.
    """
    _clear_grid_legend(grid)

    handles = image_set_legend_handles(palette)
    settings = {
        "loc": "lower center",
        "ncol": len(handles) if ncol is None else ncol,
        "frameon": False,
        "bbox_to_anchor": (x, y),
    }
    settings.update(kwargs)
    legend = grid.figure.legend(handles=handles, title=title, **settings)
    grid._legend = legend
    return legend


def place_legend_inside(grid, palette=None, panel=0, title="Image set", **kwargs):
    """Put the legend inside one of the panels, costing no figure space at all.

    panel selects the axis by position in grid.axes.flat. Use this when even
    the extra height of place_legend_below is unwelcome, keeping in mind that
    it overlaps the data region.
    """
    _clear_grid_legend(grid)

    handles = image_set_legend_handles(palette)
    settings = {"loc": "lower right", "frameon": False, "fontsize": "small"}
    settings.update(kwargs)
    axis = grid.axes.flat[panel]
    legend = axis.legend(handles=handles, title=title, **settings)
    return legend
