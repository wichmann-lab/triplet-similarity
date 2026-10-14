import re
from pathlib import Path

import numpy as np
import pandas as pd
from cblearn.preprocessing import query_from_columns, triplets_from_oddoneout
from cblearn.utils import check_query_response


DATA_ROOT = Path("..")  # folder that contains image folders, i.e. taxonomic_images, action_images, Greebles .
DATA_DIRS = ['experiment/data', 'experiment_mid_lab/data']  # relative to DATA_ROOT


def get_object_names():
    set_names = ('action', 'greebles', 'taxonomic')
    objects = {}
    for name in set_names:
        df = pd.read_csv(DATA_ROOT / f'resources/{name}_info.csv')
        df['image_set'] = name
        objects[name] = df
        print(f"{name}: {len(df)} images")
    return objects


ORIGINAL_ORDER_FILE = 'data/original_session_order.txt'  # relative to DATA_ROOT


def read_order_file(path):
    """Read a file listing session files in a fixed order, one path per line."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]


def order_sessions(data_files, order, data_root=DATA_ROOT):
    """Put data_files in the order given by `order` (paths relative to data_root).

    Files that the order does not mention keep their sorted position at the
    end, so that data added later cannot silently change the original order.
    Files in the order that do not exist are ignored.
    """
    rank = {name: index for index, name in enumerate(order)}
    known, unknown = [], []
    for path in data_files:
        name = str(Path(path).relative_to(data_root))
        (known if name in rank else unknown).append(path)
    known.sort(key=lambda path: rank[str(Path(path).relative_to(data_root))])
    return known + sorted(unknown)


def find_sessions(data_folders=DATA_DIRS, data_root=DATA_ROOT, order_file=ORIGINAL_ORDER_FILE):
    """List the session files in the order given by `order_file`, or sorted.

    The order decides how the triplets of the sessions are stacked, and
    therefore the cross-validation folds and the one-of-two selection. By
    default the files come in the order in which the file system listed them
    when the published analysis was run (`ORIGINAL_ORDER_FILE`, relative to
    data_root), so that the analysis gets the same inputs on any machine.
    With order_file=None the files are sorted by path.
    """
    # glob returns files in the order of the file system, which differs
    # between machines, so the listing is sorted before anything else
    data_files = sorted(sum((list((data_root / folder).glob('*.csv')) for folder in data_folders), start=[]))
    if order_file is not None:
        data_files = order_sessions(data_files, read_order_file(data_root / order_file), data_root)
    file_match = re.compile(r'(?P<task>[^_]+)_(?P<observer>[^_]+)_(?P<session>[^_]+)_object_triplets_(?P<date>.+)_(?P<lastblock>[^_]+).csv')
    session_df = pd.DataFrame.from_records(re.match(file_match, f.name).groupdict() for f in data_files)
    session_df['file'] = [str(f) for f in data_files]
    return session_df


def preprocess_sessions(session_df):
    session_df = session_df[~session_df.observer.str.startswith('T')]  # remove test participants
    return session_df


def load_session_data(file, drop_training=True, clean=True):
    file_df = pd.read_csv(file)
    if clean:
        # session files were split into parts, because the experiment crashed
        file_df.session = file_df.session.replace({'1b2': '1b', '5a2': '5a'})
        # drop all duplicates of triplet_index and image_set (the first element of every block contains only the break timing)
        file_df.drop_duplicates(subset=["triplet_index", "image_set"], keep = "last", inplace = True)
        # Observer O3 flipped obviously the buttons in session 1e: ~0.2% response consistency with theirself and others
        mask = (file_df.participant == 'O3') & (file_df.session == '1e')
        file_df['resp.key'] = file_df['resp.key'].mask(mask, file_df['resp.key'][mask].map({'green': 'red', 'red': 'green'}))
    if drop_training:
        file_df = file_df[file_df['block_number'] > 0]
    data = []
    groups = ['experiment', 'participant', 'session', 'date', 'image_set']
    for ix, (grp_ix, grp) in enumerate(file_df.groupby(groups)):
        d = {}
        for k, v in zip(groups, grp_ix):
            d[k] = v
        resp_df = grp[grp['resp.key'].notna()]
        if d['experiment'] == 'classic':
            raw_triplets = resp_df[['middle_index', 'image1_index', 'image2_index']].values
            responses = resp_df['resp.key']
            responses = responses.map({'green': 1, 'red': -1}).values
            responses *= np.where(resp_df['is_left'].values == 1, 1, -1)
            triplets = check_query_response(raw_triplets, responses, 'list-order')
        elif d['experiment'] == 'odd one out':
            raw_triplets = resp_df[['image1_index', 'middle_index', 'image2_index']].values
            responses = np.array(resp_df['resp.key'].values, copy=True)
            responses[resp_df['is_left'] == 1] = resp_df['resp.key'][resp_df['is_left'] == 1].map({'green': 0, 'yellow': 1, 'red': 2})
            responses[resp_df['is_left'] == 2] = resp_df['resp.key'][resp_df['is_left'] == 2].map({'green': 2, 'yellow': 1, 'red': 0})
            triplets = triplets_from_oddoneout(raw_triplets, responses.astype(int))
        else:
            raise NotImplementedError
        # trials without a recorded response (there was no time limit)
        d['n_timeout'] = np.sum(grp['resp.key'].isna())
        d['responses'] = responses
        d['raw_triplets'] = raw_triplets
        d['raw_responses'] = resp_df['resp.key'].values
        d['rt'] = resp_df['resp.rt']
        d['triplet_index'] = np.array(resp_df['triplet_index'].values, dtype=int)
        d['embedding_ready_triplets'] = np.array(triplets)
        d['n_triplets'] = len(triplets)
        d['n_trials'] = len(responses)
        d['mean_rt'] = resp_df['resp.rt'].mean()
        d['block'] = np.array(resp_df['block_number'].values, dtype=int)
        d['file'] = str(file)

        data.append(d)
    session_df = pd.DataFrame.from_records(data)
    session_df['triplet_set'] = session_df.session.str[0]
    session_df['shuffle'] = session_df.session.str[1]
    session_df['date'] = pd.to_datetime(
        # date is a string, e.g. 2023-01-14_12h01.35.719
        session_df.date, format='%Y-%m-%d_%Hh%M.%S.%f').dt.tz_localize("Europe/Berlin")
    #print(session_df.date)
    if clean:
        # we use different column names
        session_df = session_df.rename({'participant': 'observer', 'experiment': 'task'}, axis='columns')
    session_df.sort_values(['image_set', 'observer', 'task', 'triplet_set', 'shuffle'])
    return session_df


def load_all(data_folders=DATA_DIRS, data_root=DATA_ROOT, clean=True,
             fill_missing_O_by_P=False, order_file=ORIGINAL_ORDER_FILE):
    session_df = find_sessions(data_folders, data_root, order_file)
    session_df = preprocess_sessions(session_df)
    triplet_df = pd.concat([load_session_data(f, clean=True) for f in session_df['file']], axis='rows', ignore_index=True)

    if fill_missing_O_by_P:
        for Px, Ox in (('P1', 'O5'), ('P2', 'O6')):
            session_mask = triplet_df.session.isin(('1a', '1b', '2a', '2b'))
            observer_mask = triplet_df.observer == Px
            triplet_df['observer'] = triplet_df['observer'].mask(session_mask & observer_mask, Ox)
            triplet_df.drop(triplet_df[triplet_df.observer == Px].index, inplace=True)


    return triplet_df
