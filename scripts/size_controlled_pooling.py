"""Pooled data of five observers matched in size to one observer's data.

The pooled O1 - O5 condition of Step 1 (pooled-5x) stacks all sessions of five
observers, so it holds five times the data of one observer. This module builds
three pooled datasets that separate the effect of mixing observers from the
effect of having more data. Each holds one more property fixed than the one
before it.

Five observers answered two trial lists (trial sets 1 and 2, 300 rows each) in
three sessions (repeats) each. A row is a `triplet_index`: one question in one
screen layout. The lists ask some questions in two or three layouts, and every
design keeps those as separate rows, as in every observer's data. A row that an
observer did not answer in a session (timeout) is missing, never replaced.

Design A, scaled-down pool
--------------------------
`subsample_sessions` keeps a random fraction of the trials of every session of
pooled-5x. With fraction 1/5 the result is pooled-5x scaled down by five: about
the size of one observer's data, with the same observers, rows, layouts and
same-observer repeats, only fewer of each. It differs from pooled-5x in the
amount of data only, and from the individual observers in the mixing only.
The notebook `pooled_design_A_scaled_down_pool.ipynb` runs it.

Design B, crowd-like pool
-------------------------
`pool_condition` builds pooled data with the structure of one observer's data
(every row of both trial lists once per repeat, three repeats, 1800 trials), in
which the three answers to a row come from three different observers, so no
observer answers a row twice, as in crowd-sourced data.

- For every trial list, the rows are split into 10 groups, one per combination
  of 3 of the 5 observers. `REPEAT_TABLE` says which observer of each
  combination answers the group in repeat 0, 1 and 2. In every repeat each
  observer answers exactly 2 groups, so each synthetic repeat asks every row
  once, every observer answers a fifth of it, and every pair of observers
  shares 3 of the 10 groups.
- The answers of an observer in repeat r come from its r-th session of the
  trial list, in an order of its three sessions drawn at random per draw.
- A cyclic Latin square (observer i takes fifth (i + r) mod 5 in repeat r) is
  not used: it lets only 5 of the 10 observer combinations occur, and observer
  pairs would share 2 or 1 fifths depending on their position in the cycle.

`check_pooling` verifies these rules. The notebook
`pooled_design_B_crowd_like_pool.ipynb` runs the design.

Design C, matched one-rep datasets
----------------------------------
`one_rep_dataset` builds a one-rep dataset: one session of trial set 1 and one
of trial set 2, 600 rows, every row of both trial lists once, a third of one
observer's data. A combination (i, j) takes the i-th session of trial set 1 and
the j-th of trial set 2, in recording order. Single obs is one observer's two
sessions as recorded; mix obs splits the rows of each trial set at random into
five equal groups, one per observer, and takes every answer from that
observer's session of the combination. Both variants have the same size, rows
and sessions, so comparing them isolates the mixing of observers. The notebook
`pooled_design_C_matched_one_rep_datasets.ipynb` runs them.
"""

import numpy as np
import pandas as pd
from cblearn.preprocessing import triplets_from_oddoneout
from cblearn.utils import check_query_response

# Rows: the 10 combinations of 3 out of 5 observer labels. Columns: the
# observer label that answers the combination's questions in repeat 0, 1 and 2.
# In every column each label appears exactly twice.
REPEAT_TABLE = np.array([
    (0, 1, 2),
    (0, 1, 3),
    (1, 0, 4),
    (2, 0, 3),
    (2, 4, 0),
    (3, 4, 0),
    (1, 3, 2),
    (4, 2, 1),
    (4, 3, 1),
    (3, 2, 4),
])
N_OBSERVERS = 5
N_REPEATS = REPEAT_TABLE.shape[1]


def pool_condition(case_df, observers, rng):
    """Build the three synthetic repeats of design B for one task and image set.

    Parameters
    ----------
    case_df : pandas.DataFrame
        One row per session of this task and image set (as returned by
        `load_all`), both trial sets, with columns `observer`, `session`,
        `triplet_set`, `triplet_index`, `raw_triplets` and `responses`. Every
        observer must have exactly 3 sessions of every trial set.
    observers : sequence of str
        The 5 observers to pool.
    rng : numpy.random.Generator
        Source of all randomness, drawn separately for every trial set: the
        split of the rows into the groups of `REPEAT_TABLE`, the mapping of the
        observers to its labels, and the order of every observer's sessions
        over the repeats.

    Returns
    -------
    list of dict
        One dict per synthetic repeat, with keys `repetition`, `raw_triplets`
        (int), `responses`, `triplet_index`, `triplet_set`, `source_observer`
        and `source_session`, the last six with one entry per trial, sorted by
        trial set and row. A row that the assigned observer did not answer in
        its session (timeout) is missing, not replaced.
    """
    observers = list(observers)
    if len(observers) != N_OBSERVERS:
        raise ValueError(f"need {N_OBSERVERS} observers, got {len(observers)}")

    parts = [{'raw_triplets': [], 'responses': [], 'triplet_index': [], 'triplet_set': [],
              'source_observer': [], 'source_session': []} for _ in range(N_REPEATS)]
    for triplet_set in sorted(case_df['triplet_set'].unique()):
        in_set = case_df[case_df['triplet_set'] == triplet_set]
        # the rows of the trial list, split at random into one group per combination
        rows = np.unique(np.concatenate([np.asarray(i, dtype=int) for i in in_set['triplet_index']]))
        groups = np.array_split(rng.permutation(rows), len(REPEAT_TABLE))
        # which real observer carries which label of REPEAT_TABLE
        label_to_observer = [observers[k] for k in rng.permutation(N_OBSERVERS)]
        # which session of every observer serves which repeat
        session_order = {}
        for obs in observers:
            sessions = sorted(in_set.loc[in_set['observer'] == obs, 'session'])
            if len(sessions) != N_REPEATS:
                raise ValueError(f"{obs} has {len(sessions)} sessions of trial set {triplet_set}, need {N_REPEATS}")
            session_order[obs] = [sessions[k] for k in rng.permutation(N_REPEATS)]

        for group, labels in zip(groups, REPEAT_TABLE):
            for rep in range(N_REPEATS):
                obs = label_to_observer[labels[rep]]
                session = in_set[(in_set['observer'] == obs) & (in_set['session'] == session_order[obs][rep])].iloc[0]
                # select by row identity; the position in the session is shuffled
                mask = np.isin(np.asarray(session['triplet_index'], dtype=int), group)
                n = int(mask.sum())
                parts[rep]['raw_triplets'].append(np.asarray(session['raw_triplets'])[mask].astype(int).reshape(-1, 3))
                parts[rep]['responses'].append(np.asarray(session['responses'])[mask])
                parts[rep]['triplet_index'].append(np.asarray(session['triplet_index'], dtype=int)[mask])
                parts[rep]['triplet_set'].append(np.full(n, triplet_set, dtype=object))
                parts[rep]['source_observer'].append(np.full(n, obs, dtype=object))
                parts[rep]['source_session'].append(np.full(n, session['session'], dtype=object))

    repeats = []
    for rep, part in enumerate(parts):
        record = {'raw_triplets': np.vstack(part['raw_triplets'])}
        for key in ('responses', 'triplet_index', 'triplet_set', 'source_observer', 'source_session'):
            record[key] = np.concatenate(part[key])
        # the same order in every repeat: by trial set, then by row of the trial list
        order = np.lexsort((record['triplet_index'], record['triplet_set'].astype(str)))
        repeats.append({'repetition': rep, **{key: value[order] for key, value in record.items()}})
    return repeats


def subsample_sessions(case_df, fraction, rng):
    """Keep a random fraction of the trials of every session and stack them (design A).

    Parameters
    ----------
    case_df : pandas.DataFrame
        One row per session (as returned by `load_all`), with columns
        `observer`, `session`, `triplet_index`, `raw_triplets` and `responses`.
    fraction : float
        Share of the trials each session keeps: `round(fraction * n)` of its
        n trials, drawn without replacement.
    rng : numpy.random.Generator
        Source of the random selection.

    Returns
    -------
    dict
        `raw_triplets` (int), `responses`, `triplet_index`, `source_observer`
        and `source_session`, one entry per kept trial, sessions stacked in
        sorted (observer, session) order.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    parts = {'raw_triplets': [], 'responses': [], 'triplet_index': [],
             'source_observer': [], 'source_session': []}
    for _, row in case_df.sort_values(['observer', 'session']).iterrows():
        n_trials = len(row['responses'])
        keep = np.sort(rng.choice(n_trials, size=round(fraction * n_trials), replace=False))
        parts['raw_triplets'].append(np.asarray(row['raw_triplets'])[keep].astype(int).reshape(-1, 3))
        parts['responses'].append(np.asarray(row['responses'])[keep])
        parts['triplet_index'].append(np.asarray(row['triplet_index'])[keep])
        parts['source_observer'].append(np.full(len(keep), row['observer'], dtype=object))
        parts['source_session'].append(np.full(len(keep), row['session'], dtype=object))
    record = {'raw_triplets': np.vstack(parts['raw_triplets'])}
    for key in ('responses', 'triplet_index', 'source_observer', 'source_session'):
        record[key] = np.concatenate(parts[key])
    return record


def one_rep_dataset(case_df, observers, combination, rng):
    """Build a one-rep dataset of design C.

    Parameters
    ----------
    case_df : pandas.DataFrame
        One row per session of one task and image set (as returned by
        `load_all`), both trial sets, with columns `observer`, `session`,
        `date`, `triplet_set`, `triplet_index`, `raw_triplets` and `responses`.
        Every observer must have exactly 3 sessions of every trial set.
    observers : sequence of str
        The observers that answer. The rows of each trial set are split at
        random into `len(observers)` groups of equal size (differing by at most
        one row), one per observer; with one observer, it answers all rows, so
        its one-rep dataset is its two sessions as recorded.
    combination : tuple of int
        The combination (i, j), 0-based: every answer to a row of trial set 1
        comes from the observer's i-th session of trial set 1 in recording
        order (by `date`), every answer to a row of trial set 2 from its j-th
        session of trial set 2.
    rng : numpy.random.Generator
        Source of the split of the rows and of which observer gets which group.

    Returns
    -------
    dict
        `raw_triplets` (int), `responses`, `triplet_index`, `triplet_set`,
        `source_observer` and `source_session`, one entry per answered row,
        sorted by trial set and row. A row that the assigned observer did not
        answer in its session (timeout) is missing, not replaced.
    """
    observers = list(observers)
    triplet_sets = sorted(case_df['triplet_set'].unique())
    if len(combination) != len(triplet_sets):
        raise ValueError(f"need one session per trial set {triplet_sets}, got {combination}")

    parts = {'raw_triplets': [], 'responses': [], 'triplet_index': [], 'triplet_set': [],
             'source_observer': [], 'source_session': []}
    for triplet_set, repeat in zip(triplet_sets, combination):
        in_set = case_df[case_df['triplet_set'] == triplet_set]
        # the rows of the trial list, split at random into one group per observer
        rows = np.unique(np.concatenate([np.asarray(i, dtype=int) for i in in_set['triplet_index']]))
        groups = np.array_split(rng.permutation(rows), len(observers))
        group_observer = [observers[k] for k in rng.permutation(len(observers))]
        for obs, group in zip(group_observer, groups):
            sessions = in_set[in_set['observer'] == obs].sort_values('date')
            if len(sessions) != 3:
                raise ValueError(f"{obs} has {len(sessions)} sessions of trial set {triplet_set}, need 3")
            session = sessions.iloc[repeat]
            # select by row identity; the position in the session is shuffled
            mask = np.isin(np.asarray(session['triplet_index'], dtype=int), group)
            n = int(mask.sum())
            parts['raw_triplets'].append(np.asarray(session['raw_triplets'])[mask].astype(int).reshape(-1, 3))
            parts['responses'].append(np.asarray(session['responses'])[mask])
            parts['triplet_index'].append(np.asarray(session['triplet_index'], dtype=int)[mask])
            parts['triplet_set'].append(np.full(n, triplet_set, dtype=object))
            parts['source_observer'].append(np.full(n, obs, dtype=object))
            parts['source_session'].append(np.full(n, session['session'], dtype=object))

    record = {'raw_triplets': np.vstack(parts['raw_triplets'])}
    for key in ('responses', 'triplet_index', 'triplet_set', 'source_observer', 'source_session'):
        record[key] = np.concatenate(parts[key])
    # the same order for every variant: by trial set, then by row of the trial list
    order = np.lexsort((record['triplet_index'], record['triplet_set'].astype(str)))
    return {key: value[order] for key, value in record.items()}


def to_triplets(task, raw_triplets, responses):
    """Turn trials into embedding-ready triplets, as `load_data` does.

    A classic trial gives one triplet, an odd-one-out trial two.
    """
    raw_triplets = np.asarray(raw_triplets).astype(int)
    if task == 'classic':
        triplets = check_query_response(raw_triplets, np.asarray(responses), 'list-order')
    elif task == 'odd one out':
        triplets = triplets_from_oddoneout(raw_triplets, np.asarray(responses).astype(int))
    else:
        raise ValueError(f"unknown task {task!r}")
    return np.asarray(triplets)


def check_pooling(pooled_df):
    """Raise if the pooled data of design B break the rules of the design.

    `pooled_df` has one row per synthetic repeat, with columns `task`,
    `image_set`, `repetition`, `triplet_set`, `triplet_index` and
    `source_observer`. Within a task and image set, it raises if a repeat asks
    a row twice or if an observer answers the same row twice.
    """
    for (task, image_set), condition in pooled_df.groupby(['task', 'image_set']):
        frames = []
        for row in condition.itertuples():
            rows = list(zip(row.triplet_set, row.triplet_index))
            if len(set(rows)) != len(rows):
                raise ValueError(f"{task} / {image_set}, repeat {row.repetition}: a row appears twice")
            frames.append(pd.DataFrame({'observer': row.source_observer, 'triplet_set': row.triplet_set,
                                        'triplet_index': row.triplet_index}))
        answers = pd.concat(frames, ignore_index=True)
        if answers.duplicated(['observer', 'triplet_set', 'triplet_index']).any():
            raise ValueError(f"{task} / {image_set}: an observer answers the same row twice")
