"""Tests for the drawing helpers of the pooling figures."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pooling_plots import draw_accuracy_panel, finish_axes


def test_finish_axes():
    fig, ax = plt.subplots()
    finish_axes(ax, [1, 2, 3], ['a', 'b', 'c'], xlabel='k', ylabel='y', title='t')
    assert ax.get_xlim() == (0.5, 3.5)
    assert [t.get_text() for t in ax.get_xticklabels()] == ['a', 'b', 'c']
    assert not ax.spines['top'].get_visible()
    plt.close(fig)


def test_accuracy_panel_marks_significant_steps():
    fig, ax = plt.subplots()
    scores = np.array([[0.70, 0.72, 0.71, 0.73], [0.75, 0.77, 0.76, 0.78]] + [[0.75, 0.77, 0.76, 0.78]] * 3)
    draw_accuracy_panel(ax, scores, [1, 2, 3, 4, 5], 2, [0.001, 0.5, 0.9, 1.0], color='k')
    # only the dashed line of the significance criterion
    vertical = [line for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert [line.get_linestyle() for line in vertical] == ['--']
    assert vertical[0].get_xdata()[0] == 2
    texts = [t.get_text() for t in ax.texts]
    # 5 accuracies and 2 numbers for each of the 4 steps
    assert len(texts) == 5 + 2 * 4
    # mean over the 4 folds: 71.5 % at 1 dimension, 76.5 % at 2; the first gain is 5 points
    assert texts[:2] == ['71.5 %', '76.5 %']
    assert '5.0 %' in texts and '0.001' in texts
    # the only step with p < 0.05 is bold: its gain of 5 points and its p-value
    bold = [t.get_text() for t in ax.texts if t.get_weight() == 'bold']
    assert bold == ['5.0 %', '0.001']
    plt.close(fig)
