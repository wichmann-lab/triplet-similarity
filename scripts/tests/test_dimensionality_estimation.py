import numpy as np
import pandas as pd
import pytest
from cblearn.datasets import make_random_triplets

import dimensionality_estimation as de


def synthetic_triplets(dim, seed, n_objects=12, size=800):
    rng = np.random.RandomState(seed)
    points = rng.normal(size=(n_objects, dim))
    return np.asarray(make_random_triplets(points, "list-order", size=size, random_state=rng))


def test_estimate_dimensionality_requires_an_int_seed():
    triplets = synthetic_triplets(1, seed=0, size=60)
    for bad in (np.random.RandomState(0), None, 1.0):
        with pytest.raises(TypeError):
            de.estimate_dimensionality(triplets, bad, test_dimensions=(1, 2), n_repeats=1, n_jobs=1)


def test_check_seed():
    assert de.check_seed(3) == 3
    assert de.check_seed(np.int64(3)) == 3
    for bad in (None, 1.0, True, np.random.RandomState(0), "0"):
        with pytest.raises(TypeError):
            de.check_seed(bad)


def test_fit_embedding_shape_and_reproducible():
    triplets = synthetic_triplets(2, seed=1)
    first = de.fit_embedding(triplets, 2, random_state=0)
    assert first.shape == (12, 2)
    np.testing.assert_array_equal(first, de.fit_embedding(triplets, 2, random_state=0))
    assert not np.array_equal(first, de.fit_embedding(triplets, 2, random_state=1))


def test_fit_embedding_explains_noise_free_triplets():
    # triplets generated without noise from a 2-D embedding are all satisfied
    # by some 2-D embedding, so a good fit gets (almost) all of them right
    triplets = synthetic_triplets(2, seed=2, size=1500)
    embedding = de.fit_embedding(triplets, 2, random_state=0)
    anchor, near, far = embedding[triplets[:, 0]], embedding[triplets[:, 1]], embedding[triplets[:, 2]]
    correct = np.linalg.norm(anchor - near, axis=1) < np.linalg.norm(anchor - far, axis=1)
    assert correct.mean() > 0.95


def test_estimate_dimensionality_single_fit():
    triplets = synthetic_triplets(2, seed=3, size=2000)
    dim_result, embedding = de.estimate_dimensionality(
        triplets, 0, test_dimensions=(1, 2, 3), n_repeats=2, n_jobs=1)
    assert dim_result.estimated_dimension == 2
    assert embedding.shape == (12, 2)
    again, embedding_again = de.estimate_dimensionality(
        triplets, 0, test_dimensions=(1, 2, 3), n_repeats=2, n_jobs=1)
    np.testing.assert_array_equal(dim_result.test_scores, again.test_scores)
    np.testing.assert_array_equal(embedding, embedding_again)


def test_soe_seed_defaults_to_the_main_seed():
    triplets = synthetic_triplets(2, seed=4, size=600)
    kwargs = dict(test_dimensions=(1, 2), n_repeats=1, n_jobs=1)
    default, embedding = de.estimate_dimensionality(triplets, 3, **kwargs)
    same, embedding_same = de.estimate_dimensionality(triplets, 3, soe_random_state=3, **kwargs)
    np.testing.assert_array_equal(default.test_scores, same.test_scores)
    np.testing.assert_array_equal(embedding, embedding_same)


def test_soe_seed_changes_only_the_initialisation():
    # a separate SOE seed must give exactly the fit of cblearn called with that
    # SOE seed and the cross-validation splits of the main seed
    from cblearn.embedding import SOE, estimate_dimensionality_cv
    triplets = synthetic_triplets(2, seed=5, size=600)
    dim_result, embedding = de.estimate_dimensionality(
        triplets, 0, test_dimensions=(1, 2), n_repeats=1, n_jobs=1, soe_random_state=1)
    estimator = SOE(1, random_state=1)
    expected = estimate_dimensionality_cv(estimator, triplets, test_dimensions=[1, 2], n_splits=5,
                                          n_repeats=1, n_jobs=1, refit=True,
                                          random_state=np.random.RandomState(0))
    np.testing.assert_array_equal(dim_result.test_scores, expected.test_scores)
    np.testing.assert_array_equal(embedding, estimator.embedding_)
    other, _ = de.estimate_dimensionality(triplets, 0, test_dimensions=(1, 2), n_repeats=1, n_jobs=1)
    assert not np.array_equal(dim_result.test_scores, other.test_scores)


def test_soe_seed_must_be_an_int():
    triplets = synthetic_triplets(1, seed=0, size=60)
    with pytest.raises(TypeError):
        de.estimate_dimensionality(triplets, 0, test_dimensions=(1, 2), n_repeats=1, n_jobs=1,
                                   soe_random_state=np.random.RandomState(1))
