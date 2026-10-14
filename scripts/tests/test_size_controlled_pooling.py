"""Tests for the size-controlled pooling of observers."""
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cblearn.preprocessing import triplets_from_oddoneout

from load_data import load_all
from size_controlled_pooling import (REPEAT_TABLE, check_pooling, one_rep_dataset, pool_condition,
                                     subsample_sessions, to_triplets)

REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVERS = ['O1', 'O2', 'O3', 'O4', 'O5']
SESSIONS = {'1': ('1a', '1c', '1e'), '2': ('2b', '2d', '2f')}

# 500 questions asked through 600 rows of the trial lists: set 1 holds rows
# 0-299, set 2 rows 300-599. Rows 0-499 ask questions 0-499; the other rows ask
# a question again in another layout:
#   rows 500-549 --> questions 0-49 (asked in set 1 and again in set 2),
#   rows 550-589 --> questions 300-339 (asked twice within set 2),
#   rows 590-599 --> questions 0-9 (a third time).
N_QUESTIONS = 500
QUESTIONS = list(itertools.combinations(range(18), 3))[:N_QUESTIONS]
ROW_QUESTION = np.concatenate([np.arange(N_QUESTIONS), np.arange(0, 50),
                               np.arange(300, 340), np.arange(0, 10)])
N_ROWS = len(ROW_QUESTION)


def images_of(row):
    """The images of a row on screen: the question, in the layout of that row."""
    first, second, third = QUESTIONS[ROW_QUESTION[row]]
    # a repeated question swaps the comparison images (the anchor stays first)
    return np.array([first, second, third] if row < N_QUESTIONS else [first, third, second])


def set_of(row):
    return '1' if row < 300 else '2'


def response_of(observer, row):
    """A response that encodes who answered which row."""
    return 1000 * OBSERVERS.index(observer) + row


def synthetic_condition(seed=0, dropped=None):
    """5 observers x 2 trial sets x 3 sessions, each session in its own order.

    `dropped` maps an observer to rows that timed out in all of its sessions.
    """
    rng = np.random.default_rng(seed)
    dropped = dropped or {}
    rows = []
    for obs in OBSERVERS:
        for triplet_set, sessions in SESSIONS.items():
            for number, session in enumerate(sessions):
                order = rng.permutation([r for r in range(N_ROWS) if set_of(r) == triplet_set])
                order = order[~np.isin(order, list(dropped.get(obs, ())))]
                rows.append({'observer': obs, 'session': session, 'triplet_set': triplet_set,
                             'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=10 * number
                                                                                + int(triplet_set)),
                             'triplet_index': order,
                             'raw_triplets': np.array([images_of(r) for r in order], dtype=float),
                             'responses': np.array([response_of(obs, r) for r in order])})
    return pd.DataFrame(rows)


# design A, scaled-down pool: a random fifth of every session of pooled-5x

def subsample_table(record):
    """The kept trials of `subsample_sessions`, one row per trial."""
    return pd.DataFrame({'observer': record['source_observer'], 'session': record['source_session'],
                         'triplet_index': record['triplet_index'], 'response': record['responses']})


def test_every_session_keeps_the_rounded_fraction_of_its_trials():
    # O1 lost every tenth row to timeouts, so its sessions are shorter
    condition = synthetic_condition(dropped={'O1': set(range(0, N_ROWS, 10))})
    n_trials = {(r.observer, r.session): len(r.responses) for r in condition.itertuples()}
    for fraction in (1 / 5, 0.3):
        table = subsample_table(subsample_sessions(condition, fraction, np.random.default_rng(0)))
        kept = table.groupby(['observer', 'session']).size().to_dict()
        assert kept == {key: round(fraction * n) for key, n in n_trials.items()}
    # 300 rows per session --> 60; a session of 270 rows --> 54
    assert kept_for(condition, 'O2', '1a') == 60
    assert kept_for(condition, 'O1', '1a') == 54


def kept_for(condition, observer, session):
    table = subsample_table(subsample_sessions(condition, 1 / 5, np.random.default_rng(1)))
    return int(((table['observer'] == observer) & (table['session'] == session)).sum())


def test_kept_trials_come_from_their_session_once_and_stay_aligned():
    condition = synthetic_condition()
    record = subsample_sessions(condition, 1 / 5, np.random.default_rng(2))
    table = subsample_table(record)
    assert not table.duplicated(['observer', 'session', 'triplet_index']).any()
    rows_of = {(r.observer, r.session): set(r.triplet_index) for r in condition.itertuples()}
    assert all(i in rows_of[o, s] for o, s, i in zip(table['observer'], table['session'], table['triplet_index']))
    np.testing.assert_array_equal(record['raw_triplets'], [images_of(i) for i in record['triplet_index']])
    assert record['raw_triplets'].dtype.kind == 'i'
    np.testing.assert_array_equal(record['responses'],
                                  [response_of(o, i) for o, i in zip(table['observer'], table['triplet_index'])])


def test_each_trial_is_kept_with_probability_one_fifth():
    condition = synthetic_condition()
    rng = np.random.default_rng(3)
    n_draws = 500
    counts = Counter()
    for _ in range(n_draws):
        table = subsample_table(subsample_sessions(condition, 1 / 5, rng))
        counts.update(zip(table['observer'], table['session'], table['triplet_index']))
    frequency = np.array([counts[r.observer, r.session, i] for r in condition.itertuples()
                          for i in r.triplet_index]) / n_draws
    # 60 of 300 per session: the mean is exactly 1/5; each trial is binomial(500, 1/5),
    # standard deviation 0.018, so no trial should be off by more than 0.12 (6.7 sd)
    assert frequency.mean() == pytest.approx(0.2)
    assert np.abs(frequency - 0.2).max() < 0.12


def test_same_observer_repeats_occur_at_the_rate_of_a_fifth_of_pooled_5x():
    # a row answered in 3 sessions is kept in each with probability 1/5, independently,
    # so it is kept at least twice with probability 3 (1/5)^2 (4/5) + (1/5)^3 = 0.104
    expected = 3 * 0.2 ** 2 * 0.8 + 0.2 ** 3
    assert expected == pytest.approx(0.104)
    condition = synthetic_condition()
    rng = np.random.default_rng(4)
    repeated = []
    for _ in range(200):
        table = subsample_table(subsample_sessions(condition, 1 / 5, rng))
        times = table.groupby(['observer', 'triplet_index']).size().reindex(
            pd.MultiIndex.from_product([OBSERVERS, range(N_ROWS)]), fill_value=0)
        repeated.append((times >= 2).mean())
    assert np.mean(repeated) == pytest.approx(expected, abs=0.005)


def test_subsample_is_reproducible_and_checks_the_fraction():
    condition = synthetic_condition()
    first = subsample_table(subsample_sessions(condition, 1 / 5, np.random.default_rng(7)))
    again = subsample_table(subsample_sessions(condition, 1 / 5, np.random.default_rng(7)))
    other = subsample_table(subsample_sessions(condition, 1 / 5, np.random.default_rng(8)))
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)
    for fraction in (0, 1.5):
        with pytest.raises(ValueError):
            subsample_sessions(condition, fraction, np.random.default_rng(0))


def test_subsample_of_the_real_data_is_one_fifth_of_pooled_5x():
    triplet_df = load_all(data_root=REPO_ROOT, fill_missing_O_by_P=True)
    triplet_df = triplet_df[triplet_df['observer'].isin(OBSERVERS)]
    rng = np.random.default_rng(0)
    for (task, image_set), condition in triplet_df.groupby(['task', 'image_set']):
        assert len(condition) == 30  # 5 observers x 2 trial sets x 3 repeats
        total = condition['responses'].apply(len).sum()
        record = subsample_sessions(condition, 1 / 5, rng)
        # 30 sessions, each rounded by at most 0.5 trial
        assert abs(len(record['responses']) - total / 5) <= 15


# design B, crowd-like pool: every row answered by three different observers

def pool(seed=0, dropped=None, condition=None):
    condition = synthetic_condition(dropped=dropped) if condition is None else condition
    return pool_condition(condition, OBSERVERS, np.random.default_rng(seed))


def pooled_table(repeats):
    """All answers of the synthetic repeats, one row per trial."""
    return pd.concat([pd.DataFrame({'repetition': r['repetition'], 'observer': r['source_observer'],
                                    'session': r['source_session'], 'triplet_set': r['triplet_set'],
                                    'triplet_index': r['triplet_index'], 'response': r['responses']})
                      for r in repeats], ignore_index=True)


def as_df(repeats, task='classic'):
    return pd.DataFrame([{**r, 'task': task, 'image_set': 'action'} for r in repeats])


def test_repeat_table_is_balanced():
    combinations = [tuple(sorted(row)) for row in REPEAT_TABLE]
    # every combination of 3 of the 5 observers, once
    assert sorted(combinations) == list(itertools.combinations(range(5), 3))
    # every row names 3 different observers
    assert all(len(set(row)) == 3 for row in REPEAT_TABLE)
    # in every repeat each observer answers exactly 2 of the 10 groups
    for rep in range(3):
        assert Counter(REPEAT_TABLE[:, rep]) == {obs: 2 for obs in range(5)}


def test_each_repeat_asks_every_row_once_with_60_per_observer_and_trial_list():
    table = pooled_table(pool())
    # 600 rows, each answered once per repeat: 1800 trials, like one observer's data
    assert len(table) == 3 * N_ROWS
    for rep, condition in table.groupby('repetition'):
        assert list(condition['triplet_index']) == list(range(N_ROWS))
        # 2 groups of 300 / 10 rows per observer and trial list
        counts = condition.groupby(['triplet_set', 'observer']).size()
        assert counts.to_dict() == {(t, obs): 60 for t in SESSIONS for obs in OBSERVERS}


def test_three_different_observers_answer_every_row():
    table = pooled_table(pool(seed=1))
    assert not table.duplicated(['observer', 'triplet_index']).any()
    assert (table.groupby('triplet_index')['observer'].nunique() == 3).all()
    # 6 combinations x 30 rows per trial list, answered once each: 360 trials per observer
    assert table['observer'].value_counts().to_dict() == {obs: 360 for obs in OBSERVERS}


def test_observer_combinations_and_pairs_are_balanced_per_trial_list():
    table = pooled_table(pool(seed=2))
    for triplet_set, condition in table.groupby('triplet_set'):
        answered_by = condition.groupby('triplet_index')['observer'].apply(lambda o: tuple(sorted(o)))
        # 300 rows / 10 combinations
        assert Counter(answered_by) == {combo: 30 for combo in itertools.combinations(OBSERVERS, 3)}
        # every pair of observers shares 3 combinations x 30 rows
        shared = Counter(pair for combo in answered_by for pair in itertools.combinations(combo, 2))
        assert shared == {pair: 90 for pair in itertools.combinations(OBSERVERS, 2)}


def test_a_question_asked_in_several_layouts_keeps_all_its_rows():
    # question 0 is asked through rows 0, 500 and 590: all three stay, each answered three times
    table = pooled_table(pool(seed=3))
    assert table.query('triplet_index in [0, 500, 590]').groupby('triplet_index').size().to_dict() == \
        {0: 3, 500: 3, 590: 3}


def test_answers_stay_with_their_row_observer_and_session():
    for r in pool(seed=3):
        np.testing.assert_array_equal(r['raw_triplets'], [images_of(i) for i in r['triplet_index']])
        assert r['raw_triplets'].dtype.kind == 'i'
        np.testing.assert_array_equal(r['responses'],
                                      [response_of(o, i) for o, i in zip(r['source_observer'], r['triplet_index'])])
        assert all(session in SESSIONS[set_of(i)] for session, i in zip(r['source_session'], r['triplet_index']))


def test_every_session_serves_one_repeat():
    table = pooled_table(pool(seed=4))
    sessions = table.groupby(['observer', 'triplet_set', 'repetition'])['session'].unique()
    assert all(len(s) == 1 for s in sessions)
    for obs in OBSERVERS:
        for triplet_set, expected in SESSIONS.items():
            assert sorted(s[0] for s in sessions[obs, triplet_set]) == sorted(expected)


def test_the_order_of_the_sessions_is_drawn_at_random():
    condition = synthetic_condition()
    orders = {tuple(pooled_table(pool(seed=seed, condition=condition)).query("observer == 'O1' and triplet_set == '1'")
                    .groupby('repetition')['session'].first()) for seed in range(80)}
    # all 3! = 6 orders occur (each with probability 1/6; all occur in 80 draws but with
    # probability below 1e-5)
    assert orders == set(itertools.permutations(SESSIONS['1']))


def test_crowd_like_pool_leaves_timed_out_rows_missing():
    dropped = set(range(0, N_ROWS, 10))
    repeats = pool(seed=5, dropped={'O1': dropped})
    table = pooled_table(repeats)
    assert not table.query("observer == 'O1'")['triplet_index'].isin(dropped).any()
    times = table.groupby('triplet_index').size().reindex(range(N_ROWS), fill_value=0)
    # a row O1 did not time out on keeps its three answers
    assert (times[[i for i in range(N_ROWS) if i not in dropped]] == 3).all()
    # a dropped row loses its answer only if O1 is in its combination: 2 answers, otherwise 3
    assert set(times[sorted(dropped)]) <= {2, 3}
    assert 0 < (times[sorted(dropped)] == 2).sum() < len(dropped)
    check_pooling(as_df(repeats))


def test_same_seed_same_pooling_other_seed_other_pooling():
    condition = synthetic_condition()
    first = pooled_table(pool(seed=7, condition=condition))
    again = pooled_table(pool(seed=7, condition=condition))
    other = pooled_table(pool(seed=8, condition=condition))
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)


def test_pool_condition_rejects_wrong_input():
    condition = synthetic_condition()
    with pytest.raises(ValueError):
        pool_condition(condition, OBSERVERS[:4], np.random.default_rng(0))
    with pytest.raises(ValueError):
        pool_condition(condition.iloc[1:], OBSERVERS, np.random.default_rng(0))


def test_check_pooling_rejects_broken_pooling():
    row = {'task': 'classic', 'image_set': 'action'}
    good = pd.DataFrame([
        {**row, 'repetition': 0, 'triplet_set': np.array(['1', '1']), 'triplet_index': np.array([0, 1]),
         'source_observer': np.array(['O1', 'O2'])},
        {**row, 'repetition': 1, 'triplet_set': np.array(['1', '1']), 'triplet_index': np.array([0, 1]),
         'source_observer': np.array(['O2', 'O1'])},
    ])
    check_pooling(good)
    # O1 answers row 0 in both repeats
    bad = good.copy()
    bad.at[1, 'source_observer'] = np.array(['O1', 'O2'])
    with pytest.raises(ValueError):
        check_pooling(bad)
    # row 0 twice within one repeat
    bad = good.copy()
    bad.at[0, 'triplet_index'] = np.array([0, 0])
    with pytest.raises(ValueError):
        check_pooling(bad)


def test_crowd_like_pool_of_the_real_data():
    triplet_df = load_all(data_root=REPO_ROOT, fill_missing_O_by_P=True)
    triplet_df = triplet_df[triplet_df['observer'].isin(OBSERVERS)]
    rng = np.random.default_rng(0)
    records = []
    for (task, image_set), condition in triplet_df.groupby(['task', 'image_set']):
        repeats = pool_condition(condition, OBSERVERS, rng)
        # 1800 trials, fewer only by the rows the assigned observers did not answer in their sessions
        # (timeouts; O4's first classic session of trial set 1 has 150 of its 300 rows)
        assert 1740 <= sum(len(r['responses']) for r in repeats) <= 1800
        records.extend({**r, 'task': task, 'image_set': image_set} for r in repeats)
    pooled = pd.DataFrame(records)
    assert len(pooled) == 2 * 3 * 3  # tasks x image sets x repeats
    check_pooling(pooled)


# design C, matched one-rep datasets: one session of each trial list

def session_table(record):
    """The rows of a one-rep dataset, one row per trial."""
    return pd.DataFrame({'observer': record['source_observer'], 'session': record['source_session'],
                         'triplet_set': record['triplet_set'], 'triplet_index': record['triplet_index'],
                         'response': record['responses']})


def build(observers=OBSERVERS, combination=(0, 0), seed=0, condition=None):
    condition = synthetic_condition() if condition is None else condition
    return one_rep_dataset(condition, observers, combination, np.random.default_rng(seed))


def test_a_one_rep_dataset_asks_every_row_of_both_trial_lists_once():
    table = session_table(build())
    # 300 rows of trial set 1 and 300 of trial set 2, each once, sorted by set and row
    assert list(table['triplet_index']) == list(range(N_ROWS))
    assert list(table['triplet_set']) == [set_of(r) for r in range(N_ROWS)]


def test_every_observer_answers_a_fifth_of_each_trial_set():
    table = session_table(build(seed=1))
    # 300 / 5 = 60 rows per observer and trial set
    counts = table.groupby(['triplet_set', 'observer']).size()
    assert counts.to_dict() == {(t, obs): 60 for t in SESSIONS for obs in OBSERVERS}
    # 300 / 3: group sizes differ by at most one row
    table = session_table(build(observers=OBSERVERS[:3], seed=1))
    assert sorted(table.groupby(['triplet_set', 'observer']).size()) == [100] * 6


def test_answers_come_from_the_session_of_the_combination():
    # the sessions of a trial set in recording order: SESSIONS gives them in label order
    for i, j in [(0, 0), (1, 2), (2, 1)]:
        table = session_table(build(combination=(i, j), seed=2))
        expected = {'1': SESSIONS['1'][i], '2': SESSIONS['2'][j]}
        assert (table['session'] == table['triplet_set'].map(expected)).all()


def test_recording_order_comes_from_the_date_not_the_label():
    condition = synthetic_condition()
    # O2 recorded its trial set 1 sessions in the order 1e, 1a, 1c
    shift = {'1a': 20, '1c': 30, '1e': 0}
    mask = (condition['observer'] == 'O2') & (condition['triplet_set'] == '1')
    condition.loc[mask, 'date'] = [pd.Timestamp('2023-01-01') + pd.Timedelta(days=shift[s])
                              for s in condition.loc[mask, 'session']]
    table = session_table(build(observers=['O2'], combination=(0, 1), condition=condition))
    assert set(table.query("triplet_set == '1'")['session']) == {'1e'}
    assert set(table.query("triplet_set == '2'")['session']) == {'2d'}


def test_answers_stay_with_their_row_and_observer():
    record = build(seed=3)
    np.testing.assert_array_equal(record['raw_triplets'], [images_of(i) for i in record['triplet_index']])
    assert record['raw_triplets'].dtype.kind == 'i'
    np.testing.assert_array_equal(record['responses'],
                                  [response_of(o, i) for o, i in zip(record['source_observer'],
                                                                     record['triplet_index'])])


def test_one_observer_gives_its_two_sessions_as_recorded():
    condition = synthetic_condition()
    record = build(observers=['O4'], combination=(2, 1), condition=condition)
    table = session_table(record)
    assert set(table['observer']) == {'O4'}
    for session in ('1e', '2d'):
        recorded = condition[(condition['observer'] == 'O4') & (condition['session'] == session)].iloc[0]
        kept = table[table['session'] == session]
        # all rows of the recorded session, with its responses
        order = np.argsort(recorded['triplet_index'])
        np.testing.assert_array_equal(kept['triplet_index'], np.asarray(recorded['triplet_index'])[order])
        np.testing.assert_array_equal(kept['response'], np.asarray(recorded['responses'])[order])


def test_each_observer_answers_a_row_with_probability_one_fifth():
    condition = synthetic_condition()
    rng = np.random.default_rng(4)
    n_draws = 200
    counts = Counter()
    for _ in range(n_draws):
        table = session_table(one_rep_dataset(condition, OBSERVERS, (0, 0), rng))
        counts.update(zip(table['triplet_index'], table['observer']))
    frequency = np.array([[counts[r, obs] for obs in OBSERVERS] for r in range(N_ROWS)]) / n_draws
    # every draw gives each observer exactly 120 of the 600 rows, so the mean is exactly 1/5;
    # each cell is binomial(200, 1/5), sd 0.028, so none should be off by more than 0.17 (6 sd)
    assert frequency.mean() == pytest.approx(0.2)
    assert np.abs(frequency - 0.2).max() < 0.17


def test_mixed_and_single_one_rep_datasets_have_the_same_rows():
    condition = synthetic_condition()
    mixed = session_table(build(combination=(1, 1), seed=5, condition=condition))
    for obs in OBSERVERS:
        single = session_table(build(observers=[obs], combination=(1, 1), condition=condition))
        np.testing.assert_array_equal(single['triplet_index'], mixed['triplet_index'])


def test_one_rep_dataset_leaves_timed_out_rows_missing():
    dropped = set(range(0, N_ROWS, 10))
    condition = synthetic_condition(dropped={'O1': dropped})
    table = session_table(build(seed=6, condition=condition))
    assert not table.query("observer == 'O1'")['triplet_index'].isin(dropped).any()
    # only rows that O1 timed out on can be missing, and they are not given to another observer
    missing = set(range(N_ROWS)) - set(table['triplet_index'])
    assert missing <= dropped
    single = session_table(build(observers=['O1'], condition=condition))
    assert set(range(N_ROWS)) - set(single['triplet_index']) == dropped


def test_one_rep_dataset_is_reproducible_and_checks_its_input():
    condition = synthetic_condition()
    first = session_table(build(seed=7, condition=condition))
    again = session_table(build(seed=7, condition=condition))
    other = session_table(build(seed=8, condition=condition))
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)
    with pytest.raises(ValueError):
        build(combination=(0,), condition=condition)
    with pytest.raises(ValueError):
        build(condition=condition[~((condition['observer'] == 'O1') & (condition['session'] == '1a'))])


def test_one_rep_datasets_of_the_real_data():
    triplet_df = load_all(data_root=REPO_ROOT, fill_missing_O_by_P=True)
    triplet_df = triplet_df[triplet_df['observer'].isin(OBSERVERS)]
    rng = np.random.default_rng(0)
    for (task, image_set), condition in triplet_df.groupby(['task', 'image_set']):
        # recording order by date is the order of the session labels, for every observer
        for (obs, triplet_set), sessions in condition.groupby(['observer', 'triplet_set']):
            assert list(sessions.sort_values('date')['session']) == sorted(sessions['session'])
        for combination in itertools.product(range(3), repeat=2):
            mixed = one_rep_dataset(condition, OBSERVERS, combination, rng)
            # every row at most once, at most 2 x 300 rows
            assert len(set(zip(mixed['triplet_set'], mixed['triplet_index']))) == len(mixed['responses']) <= 600
            # a row is missing only if some observer's session of the combination lacks it
            # (timeouts, and the classic sessions 1b of O4, stopped after 150 trials)
            for triplet_set, repeat in zip(sorted(condition['triplet_set'].unique()), combination):
                in_set = condition[condition['triplet_set'] == triplet_set]
                all_rows = set(np.concatenate([np.asarray(i, dtype=int) for i in in_set['triplet_index']]))
                sessions = [set(np.asarray(s.sort_values('date').iloc[repeat]['triplet_index'], dtype=int))
                            for _, s in in_set.groupby('observer')]
                kept = set(mixed['triplet_index'][mixed['triplet_set'] == triplet_set])
                assert all_rows - kept <= all_rows - set.intersection(*sessions)
            triplets = to_triplets(task, mixed['raw_triplets'], mixed['responses'])
            assert len(triplets) == (1 if task == 'classic' else 2) * len(mixed['responses'])


# triplets, shared by the designs

def test_to_triplets_classic():
    raw = np.array([[0, 1, 2], [0, 1, 2]])
    # response 1: the first image is closer to the second; -1: closer to the third
    np.testing.assert_array_equal(to_triplets('classic', raw, np.array([1, -1])),
                                  [[0, 1, 2], [0, 2, 1]])


def test_to_triplets_odd_one_out():
    raw = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    responses = np.array([0, 2], dtype=object)
    triplets = to_triplets('odd one out', raw, responses)
    # two triplets per trial: each of the similar pair is closer to the other than to the odd one
    assert len(triplets) == 2 * len(raw)
    # cblearn lists the first triplet of every trial, then the second ones
    assert sorted(map(tuple, triplets.tolist())) == [(1, 2, 0), (2, 1, 0), (3, 4, 5), (4, 3, 5)]
    np.testing.assert_array_equal(triplets, triplets_from_oddoneout(raw.astype(int), responses.astype(int)))
    with pytest.raises(ValueError):
        to_triplets('odd-one-out-oneoftwo', raw, responses)
