import numpy as np
import pandas as pd
import pytest
from cblearn.datasets import make_random_triplets

import dimensionality_estimation as de
import seed_stability as ss


def synthetic(dim=2, seed=0, n_objects=10, size=600):
    rng = np.random.RandomState(seed)
    points = rng.normal(size=(n_objects, dim))
    return np.asarray(make_random_triplets(points, "list-order", size=size, random_state=rng))


def test_seeds_for():
    assert ss.seeds_for("both", 3) == (3, 3)
    assert ss.seeds_for("soe", 3) == (3, 0)
    assert ss.seeds_for("cv", 3) == (0, 3)
    assert ss.seeds_for("soe", 3, base=7) == (3, 7)
    with pytest.raises(ValueError):
        ss.seeds_for("neither", 3)
    with pytest.raises(TypeError):
        ss.seeds_for("both", 1.5)


def test_fit_once_is_reproducible_and_seed_dependent():
    triplets = synthetic()
    kwargs = dict(test_dimensions=(1, 2), n_repeats=1, n_jobs=1)
    first = ss.fit_once(triplets, 0, 0, **kwargs)
    again = ss.fit_once(triplets, 0, 0, **kwargs)
    np.testing.assert_array_equal(first.test_scores, again.test_scores)
    # the cross-validation seed changes which triplets land in which fold
    other_cv = ss.fit_once(triplets, 0, 1, **kwargs)
    assert not np.array_equal(first.test_scores, other_cv.test_scores)


def test_fit_once_is_the_fit_of_the_analysis():
    # a stability run is the fit the notebooks make with the same two seeds
    triplets = synthetic(seed=1)
    kwargs = dict(test_dimensions=(1, 2), n_repeats=1, n_jobs=1)
    expected, _ = de.estimate_dimensionality(triplets, 2, soe_random_state=5, **kwargs)
    result = ss.fit_once(triplets, soe_seed=5, cv_seed=2, **kwargs)
    np.testing.assert_array_equal(result.test_scores, expected.test_scores)


def test_stability_runs_shape_and_labels():
    triplets = synthetic()
    runs = ss.stability_runs(triplets, [0, 1], "soe", label={"observer": "O1"},
                             test_dimensions=(1, 2), n_repeats=1, n_jobs=1)
    assert list(runs["vary"]) == ["soe", "soe"]
    assert list(runs["soe_seed"]) == [0, 1] and list(runs["cv_seed"]) == [0, 0]
    assert list(runs["observer"]) == ["O1", "O1"]
    assert "p_1to2" in runs.columns
    assert runs["dimensionality"].between(1, 2).all()


def test_summarise():
    runs = pd.DataFrame({
        "observer": ["O1"] * 3 + ["O2"] * 3,
        "task": ["classic"] * 6, "image_set": ["action"] * 6,
        "vary": ["soe"] * 6,
        "dimensionality": [2, 2, 2, 1, 2, 2],
    })
    table = ss.summarise(runs).set_index("observer")
    assert not table.loc["O1", "unstable"] and table.loc["O1", "n_distinct"] == 1
    assert table.loc["O2", "unstable"] and table.loc["O2", "dim_min"] == 1
    assert table.loc["O2", "dim_max"] == 2 and table.loc["O2", "dim_mode"] == 2
    assert table.loc["O1", "n_runs"] == 3


def soe_runs(observer, dims, cv_seed=0):
    return pd.DataFrame({"observer": observer, "task": "odd one out", "image_set": "action",
                         "vary": "soe", "seed": range(len(dims)), "soe_seed": range(len(dims)),
                         "cv_seed": cv_seed, "dimensionality": dims})


def test_majority_rule_picks_the_majority_and_its_lowest_seed():
    runs = pd.concat([soe_runs("P1", [1, 2, 1, 2, 2, 2, 2, 2, 2, 2]),
                      soe_runs("O4", [1, 1, 1, 1, 1, 1, 1, 1, 2, 1])])
    table = ss.majority_rule(runs).set_index("observer")
    # P1: 2 in 8 of 10 seeds, first given by seed 1
    assert table.loc["P1", "dimensionality"] == 2
    assert table.loc["P1", "share"] == pytest.approx(0.8)
    assert table.loc["P1", "soe_seed"] == 1
    # O4: 1 in 9 of 10 seeds, already at seed 0
    assert table.loc["O4", "dimensionality"] == 1
    assert table.loc["O4", "share"] == pytest.approx(0.9)
    assert table.loc["O4", "soe_seed"] == 0


def test_majority_rule_breaks_ties_towards_the_lower_dimensionality():
    table = ss.majority_rule(soe_runs("O1", [2, 1, 2, 1]))
    assert table.loc[0, "dimensionality"] == 1 and table.loc[0, "soe_seed"] == 1
    assert table.loc[0, "share"] == pytest.approx(0.5)


def test_majority_rule_uses_only_soe_runs_at_the_given_cv_seed():
    runs = pd.concat([soe_runs("O1", [2, 2, 2]),
                      soe_runs("O1", [1, 1, 1], cv_seed=1),
                      soe_runs("O1", [1, 1, 1]).assign(vary="cv")])
    table = ss.majority_rule(runs)
    assert table.loc[0, "dimensionality"] == 2 and table.loc[0, "n_seeds"] == 3
    with pytest.raises(ValueError):
        ss.majority_rule(runs, cv_seed=5)


def test_committed_results_give_the_override_of_the_dimensionality_notebook():
    # finding_dimensionality_aggregated_per_observer.ipynb hard-codes
    # SOE_SEED_OVERRIDE = {('P1', 'odd one out', 'action'): 1}; it must stay
    # what the majority rule gives on the committed seed-stability runs
    from pathlib import Path
    results = Path(__file__).resolve().parents[2] / "data" / Path(ss.RESULTS_FILE).name
    table = ss.majority_rule(pd.read_csv(results)).set_index(["observer", "task", "image_set"])
    expected = {("O2", "odd one out", "greebles"): (2, 0.8, 0),
                ("O4", "odd one out", "taxonomic"): (1, 0.9, 0),
                ("P1", "odd one out", "action"): (2, 0.8, 1)}
    assert set(table.index) == set(expected)
    for key, (dimensionality, share, soe_seed) in expected.items():
        assert table.loc[key, "dimensionality"] == dimensionality
        assert table.loc[key, "share"] == pytest.approx(share)
        assert table.loc[key, "soe_seed"] == soe_seed
