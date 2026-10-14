"""Seeded fits of ordinal embeddings with cblearn.

Every embedding fit in the analysis goes through this module, so that all of
them use the same settings and are reproducible. It only fits; the saved
results are read back with `utils.py` and `io_json.py`.

Seeding
-------
`random_state` is always an int. cblearn's `SOE` defaults to
`random_state=None`, which draws new initial embeddings on every fit.
With an int, `check_random_state` builds a fresh generator at the start of
every `fit`, so the clones that `estimate_dimensionality_cv` makes for each
fold and dimension, and the final refit, all start from the same initial
embeddings. A shared numpy RandomState would instead be advanced by every
fit, making a result depend on the fits run before it and, with n_jobs > 1,
on the order in which parallel folds finish. RandomState objects are
therefore rejected.

The same int also seeds the cross-validation splits.
"""

import numpy as np
from cblearn.embedding import SOE, estimate_dimensionality_cv

TEST_DIMENSIONS = (1, 2, 3, 4, 5)
N_SPLITS = 5
N_REPEATS = 30
# 10 = the physical cores of the analysis machine (i7-1355U: 2 performance cores
# with hyperthreading plus 8 efficiency cores, 12 logical CPUs). Measured: 15.1 s
# per reduced fit at 10 jobs, 17.0 s at 12 (hyperthreads do not help), 28.0 s at 6.
N_JOBS = 10


def check_seed(random_state):
    """Return random_state as an int, or raise if it is not one."""
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an int, got {0!r}".format(type(random_state)))
    return int(random_state)


def fit_embedding(triplets, n_components, random_state):
    """Fit an SOE embedding of fixed dimensionality.

    Returns
    -------
    numpy.ndarray
        The embedding, shape (n_objects, n_components).
    """
    estimator = SOE(n_components=n_components, random_state=check_seed(random_state))
    estimator.fit(np.asarray(triplets), y=None)
    return estimator.embedding_


def estimate_dimensionality(triplets, random_state, test_dimensions=TEST_DIMENSIONS,
                            n_splits=N_SPLITS, n_repeats=N_REPEATS, n_jobs=N_JOBS,
                            soe_random_state=None):
    """Estimate dimensionality by cross-validation and fit the embedding.

    Uses `cblearn.embedding.estimate_dimensionality_cv` with the significance
    criterion, then refits SOE on all triplets at the estimated dimension.

    `random_state` seeds the cross-validation splits and, unless
    `soe_random_state` is given, also the SOE initialisations. The two are
    separate ingredients of a fit; `soe_random_state` changes the
    initialisations while keeping the splits.

    Returns
    -------
    tuple
        (dim_result, embedding): the DimensionEstimationResult of cblearn and
        the refitted embedding, shape (n_objects, dim_result.estimated_dimension).
    """
    seed = check_seed(random_state)
    soe_seed = seed if soe_random_state is None else check_seed(soe_random_state)
    estimator = SOE(1, random_state=soe_seed)
    dim_result = estimate_dimensionality_cv(estimator,
                                            triplets,
                                            test_dimensions=list(test_dimensions),
                                            n_splits=n_splits,
                                            n_repeats=n_repeats,
                                            n_jobs=n_jobs,
                                            refit=True,
                                            random_state=np.random.RandomState(seed))
    assert estimator.embedding_.shape[1] == dim_result.estimated_dimension
    return dim_result, estimator.embedding_
