"""Tests for the order in which the session files are read."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_data import (DATA_DIRS, ORIGINAL_ORDER_FILE, find_sessions, load_all,
                       order_sessions, read_order_file)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATE = "2023-01-14_12h01.35.719"


def write_session(path, experiment, participant, session, rows):
    """A minimal PsychoPy csv: one row per trial plus a training trial."""
    records = [{"experiment": experiment, "participant": participant, "session": session,
                "date": DATE, "image_set": "action", "block_number": 0, "triplet_index": 99,
                "middle_index": 0, "image1_index": 1, "image2_index": 2, "is_left": 1,
                "resp.key": "green", "resp.rt": 1.0}]
    for index, (middle, image1, image2, is_left, key) in enumerate(rows):
        records.append({"experiment": experiment, "participant": participant, "session": session,
                        "date": DATE, "image_set": "action", "block_number": 1,
                        "triplet_index": index, "middle_index": middle, "image1_index": image1,
                        "image2_index": image2, "is_left": is_left, "resp.key": key,
                        "resp.rt": 1.5})
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_find_sessions_without_order_file_is_sorted_by_path(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    names = [
        "odd one out_O2_1d_object_triplets_2023-02-09_16h45.57.861_5.csv",
        "classic_O1_1a_object_triplets_2023-01-14_12h01.35.719_6.csv",
        "classic_O1_1c_object_triplets_2023-01-20_10h00.00.000_6.csv",
    ]
    # create the files in an order that is not the sorted one
    for name in (names[2], names[0], names[1]):
        (folder / name).write_text("")
    sessions = find_sessions(data_folders=["data"], data_root=tmp_path, order_file=None)
    assert [os.path.basename(f) for f in sessions["file"]] == sorted(names)
    assert list(sessions["observer"]) == ["O1", "O1", "O2"]
    assert list(sessions["session"]) == ["1a", "1c", "1d"]


def test_load_all_uses_the_given_folders(tmp_path):
    folder = tmp_path / "sessions"
    folder.mkdir()
    name = "classic_O1_1a_object_triplets_{0}_6.csv".format(DATE)
    write_session(folder / name, "classic", "O1", "1a", [(0, 1, 2, 1, "green")])
    triplet_df = load_all(data_folders=["sessions"], data_root=tmp_path, order_file=None)
    assert list(triplet_df["observer"]) == ["O1"]
    np.testing.assert_array_equal(triplet_df.iloc[0]["embedding_ready_triplets"], [[0, 1, 2]])


def test_order_sessions_follows_the_given_order(tmp_path):
    names = ["b.csv", "a.csv", "c.csv"]
    files = [tmp_path / "data" / n for n in names]
    order = ["data/c.csv", "data/a.csv", "data/b.csv"]
    ordered = order_sessions(sorted(files), order, data_root=tmp_path)
    assert [f.name for f in ordered] == ["c.csv", "a.csv", "b.csv"]


def test_order_sessions_puts_unknown_files_last_and_sorted(tmp_path):
    files = [tmp_path / "data" / n for n in ("b.csv", "a.csv", "z.csv", "new.csv")]
    order = ["data/b.csv", "data/a.csv"]
    ordered = order_sessions(files, order, data_root=tmp_path)
    assert [f.name for f in ordered] == ["b.csv", "a.csv", "new.csv", "z.csv"]


def test_order_sessions_ignores_listed_files_that_do_not_exist(tmp_path):
    files = [tmp_path / "data" / "a.csv"]
    order = ["data/gone.csv", "data/a.csv"]
    assert order_sessions(files, order, data_root=tmp_path) == files


def test_read_order_file(tmp_path):
    path = tmp_path / "order.txt"
    path.write_text("# a comment\n\nexperiment/data/b.csv\nexperiment/data/a.csv\n")
    assert read_order_file(path) == ["experiment/data/b.csv", "experiment/data/a.csv"]


def test_find_sessions_uses_the_order_file(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    names = ["classic_O1_1a_object_triplets_2023-01-14_12h01.35.719_6.csv",
             "classic_O2_1c_object_triplets_2023-01-20_10h00.00.000_6.csv"]
    for name in names:
        (folder / name).write_text("")
    (tmp_path / "order.txt").write_text("data/{0}\ndata/{1}\n".format(names[1], names[0]))
    sessions = find_sessions(data_folders=["data"], data_root=tmp_path, order_file="order.txt")
    assert list(sessions["observer"]) == ["O2", "O1"]


def test_find_sessions_reads_the_original_order_by_default(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    names = ["classic_O1_1a_object_triplets_2023-01-14_12h01.35.719_6.csv",
             "classic_O2_1c_object_triplets_2023-01-20_10h00.00.000_6.csv"]
    for name in names:
        (folder / name).write_text("")
    order_path = tmp_path / ORIGINAL_ORDER_FILE
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_text("data/{0}\ndata/{1}\n".format(names[1], names[0]))
    sessions = find_sessions(data_folders=["data"], data_root=tmp_path)
    assert list(sessions["observer"]) == ["O2", "O1"]


def test_find_sessions_fails_without_the_order_file(tmp_path):
    # a missing order file must not silently fall back to another order
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        find_sessions(data_folders=["data"], data_root=tmp_path)


def test_original_order_file_lists_exactly_the_session_files():
    order = read_order_file(REPO_ROOT / ORIGINAL_ORDER_FILE)
    on_disk = [str(path.relative_to(REPO_ROOT))
               for folder in DATA_DIRS for path in (REPO_ROOT / folder).glob("*.csv")]
    assert len(order) == len(set(order))
    assert set(order) == set(on_disk)


def test_default_order_of_the_repository_is_the_original_one():
    sessions = find_sessions(data_root=REPO_ROOT)
    listed = [str(Path(f).relative_to(REPO_ROOT)) for f in sessions["file"]]
    assert listed == read_order_file(REPO_ROOT / ORIGINAL_ORDER_FILE)
