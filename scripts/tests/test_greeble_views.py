import numpy as np
import pandas as pd
import pytest

from greeble_views import classic_same_individual, odd_one_out_same_individual, same_individual_rate

# images 0 and 1 are two views of individual 10, images 2 and 3 are different individuals
INDIVIDUAL = np.array([10, 10, 20, 30])


def test_classic_counts_only_trials_with_the_other_view():
    triplets = [
        [0, 1, 2],  # reference 0, other view judged closer: kept together
        [1, 2, 0],  # reference 1, other view judged farther: not kept together
        [2, 0, 3],  # reference has no second view: not informative
        [2, 0, 1],  # both views are candidates: not informative
    ]
    assert classic_same_individual(triplets, INDIVIDUAL) == (1, 2)


def test_classic_without_informative_trials():
    assert classic_same_individual([[2, 3, 0]], INDIVIDUAL) == (0, 0)


def test_odd_one_out_kept_together_when_the_other_greeble_is_odd():
    raw = [
        [0, 1, 2],  # odd at position 2 (image 2): views kept together
        [0, 2, 1],  # odd at position 0 (image 0, a view): split
        [2, 0, 1],  # odd at position 0 (image 2): views kept together
        [0, 2, 3],  # three individuals: not informative
    ]
    odd = [2, 0, 0, 1]
    assert odd_one_out_same_individual(raw, odd, INDIVIDUAL) == (2, 3)


def test_odd_one_out_every_position_of_the_single_image():
    # the single image at each of the three positions, always chosen as odd
    raw = [[2, 0, 1], [0, 2, 1], [0, 1, 2]]
    assert odd_one_out_same_individual(raw, [0, 1, 2], INDIVIDUAL) == (3, 3)
    # and never chosen
    assert odd_one_out_same_individual(raw, [1, 0, 0], INDIVIDUAL) == (0, 3)


def test_same_individual_rate_pools_sessions_of_an_observer():
    sessions = pd.DataFrame({
        'observer': ['O1', 'O1', 'O1'],
        'task': ['classic', 'classic', 'odd one out'],
        'embedding_ready_triplets': [np.array([[0, 1, 2]]), np.array([[1, 2, 0], [0, 1, 3]]), None],
        'raw_triplets': [None, None, np.array([[0, 1, 2], [0, 1, 3]])],
        'responses': [None, None, np.array([2, 0])],
    })
    rates = same_individual_rate(sessions, INDIVIDUAL)
    assert rates[('O1', 'classic')] == pytest.approx((100 * 2 / 3, 3))
    assert rates[('O1', 'odd one out')] == pytest.approx((50.0, 2))


def test_same_individual_rate_rejects_unknown_task():
    sessions = pd.DataFrame({'observer': ['O1'], 'task': ['ranking'],
                             'embedding_ready_triplets': [None], 'raw_triplets': [None], 'responses': [None]})
    with pytest.raises(ValueError):
        same_individual_rate(sessions, INDIVIDUAL)
