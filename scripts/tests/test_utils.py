"""Tests for the readers of the dimensionality results."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import read_dimensionality

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_result(folder, name, dimension):
    (folder / 'data').mkdir(exist_ok=True)
    with open(folder / 'data' / f'triplet_accuracy_vs_dimensionality-{name}.json', 'w') as f:
        json.dump({'estimated_dimension': dimension, 'dimensions': [1, 2, 3]}, f)


def test_read_dimensionality_picks_the_json_of_the_criterion(tmp_path):
    write_result(tmp_path, 'O1-classic-action', 3)
    write_result(tmp_path, 'O1-classic-action_LogOdds', 1)
    assert read_dimensionality('O1', 'classic', 'action', data_root=tmp_path) == 3
    assert read_dimensionality('O1', 'classic', 'action', 'significance', data_root=tmp_path) == 3
    assert read_dimensionality('O1', 'classic', 'action', 'LogOdds', data_root=tmp_path) == 1
    # returned as a plain int, whatever number type the json holds
    assert isinstance(read_dimensionality('O1', 'classic', 'action', 'LogOdds', data_root=tmp_path), int)


def test_read_dimensionality_rejects_an_unknown_criterion(tmp_path):
    write_result(tmp_path, 'O1-classic-action', 3)
    with pytest.raises(ValueError):
        read_dimensionality('O1', 'classic', 'action', 'effectsize', data_root=tmp_path)


@pytest.mark.parametrize('table', ['aggregated-per-observer', 'pooled-observers'])
def test_logodds_tables_agree_with_their_jsons(table):
    # the -LogOdds tables once kept the significance dimensionality next to the
    # LogOdds embedding; the column, the embedding and the json must agree
    df = pd.read_csv(REPO_ROOT / f'data/found-dimensionality-embeddings-{table}-nsplit-5-nrepeat-30-LogOdds.csv')
    for row in df.itertuples():
        from_json = read_dimensionality(row.observer, row.task, row.image_set, 'LogOdds', data_root=REPO_ROOT)
        assert row.dimensionality == from_json, (row.observer, row.task, row.image_set)
        assert np.array(ast.literal_eval(row.embedding)).shape[1] == from_json
