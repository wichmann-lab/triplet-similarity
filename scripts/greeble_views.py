"""How often observers group two views of the same greeble (Supplementary Table S6).

Six of the 18 greebles appear in two views (column `same_individual` of
`resources/greebles_info.csv`). A trial is informative when it contains two
images of the same individual and one image of another individual:

- classic task: the reference is one view and the other view is one of the
  two candidates; the observer picks the other view or the other greeble;
- odd-one-out task: the two views and another greeble; the observer picks the
  other greeble as odd (both views kept together) or one of the views.

The rate is the percentage of informative trials in which the two views were
kept together.
"""
import numpy as np


def classic_same_individual(triplets, individual):
    """Count informative classic trials and those where the other view was chosen.

    triplets: (n, 3) embedding-ready triplets, anchor (reference) first and the
        image judged more similar second.
    individual: array mapping image index to greeble individual.
    Returns (kept_together, n_informative).
    """
    ids = np.asarray(individual)[np.asarray(triplets, dtype=int)]
    anchor, closer, farther = ids[:, 0], ids[:, 1], ids[:, 2]
    informative = ((anchor == closer) | (anchor == farther)) & (closer != farther)
    kept_together = informative & (anchor == closer)
    return int(kept_together.sum()), int(informative.sum())


def odd_one_out_same_individual(raw_triplets, odd_positions, individual):
    """Count informative odd-one-out trials and those where both views were kept together.

    raw_triplets: (n, 3) image indices of the trials.
    odd_positions: (n,) position (0, 1 or 2) of the image chosen as odd one out.
    individual: array mapping image index to greeble individual.
    Returns (kept_together, n_informative).
    """
    ids = np.asarray(individual)[np.asarray(raw_triplets, dtype=int)]
    odd_positions = np.asarray(odd_positions, dtype=int)
    n_distinct = np.array([len(set(row)) for row in ids])
    informative = n_distinct == 2
    # the odd image is the one whose individual appears only once
    odd_id = ids[np.arange(len(ids)), odd_positions]
    odd_is_unique = (ids == odd_id[:, None]).sum(axis=1) == 1
    kept_together = informative & odd_is_unique
    return int(kept_together.sum()), int(informative.sum())


def same_individual_rate(triplet_df, individual):
    """Percentage of informative trials keeping the two views together, per observer and task.

    triplet_df: output of load_data.load_all, restricted to the greebles.
    Returns a dict {(observer, task): (percentage, n_informative)}.
    """
    counts = {}
    for _, row in triplet_df.iterrows():
        if row['task'] == 'classic':
            kept, n = classic_same_individual(row['embedding_ready_triplets'], individual)
        elif row['task'] == 'odd one out':
            kept, n = odd_one_out_same_individual(row['raw_triplets'], row['responses'], individual)
        else:
            raise ValueError(f"unknown task {row['task']}")
        key = (row['observer'], row['task'])
        previous_kept, previous_n = counts.get(key, (0, 0))
        counts[key] = (previous_kept + kept, previous_n + n)
    return {key: (100 * kept / n, n) for key, (kept, n) in counts.items()}
