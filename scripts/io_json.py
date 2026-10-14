from dataclasses import asdict
import json
import numpy as np

# Numpy dtype kinds treated as numeric and therefore restored to arrays:
# b = bool, i = signed int, u = unsigned int, f = float, c = complex.
_NUMERIC_KINDS = "biufc"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json_dataclass(dim_result, filetosave):
    with open(filetosave, "w", encoding="utf-8") as f:
        json.dump(asdict(dim_result), f, cls=NumpyEncoder)


def save_json_dict(dim_result, filetosave):
    with open(filetosave, "w", encoding="utf-8") as f:
        json.dump(dim_result, f, cls=NumpyEncoder)


def _list_to_array(value):
    """Return a numpy array if value is a rectangular, purely numeric list,
    otherwise return None to signal "not an array, handle element-wise".

    Empty lists return None on purpose: JSON carries no dtype, so there is no
    principled choice of dtype for an empty array, and a plain list is the
    safer round-trip.
    """
    if len(value) == 0:
        return None
    try:
        arr = np.array(value)
    except (ValueError, TypeError):
        # Ragged nested lists raise in modern numpy.
        return None
    if arr.dtype.kind in _NUMERIC_KINDS:
        return arr
    return None


def _restore_arrays(value):
    """Recursively turn numeric lists back into numpy arrays.

    Rules:
    - dict --> recurse into every value, keep dict
    - list --> if rectangular and purely numeric, return np.ndarray;
               otherwise recurse element-wise and keep a list
    - anything else (int, float, str, bool, None) --> returned unchanged
    """
    if isinstance(value, dict):
        return {key: _restore_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        arr = _list_to_array(value)
        if arr is not None:
            return arr
        return [_restore_arrays(item) for item in value]
    return value


def _read_json(filetoread):
    with open(filetoread, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _restore_arrays(data)


def read_json_dict(filetoread):
    """Read a JSON file saved with save_json_dict.

    Returns a dict with numeric lists restored to numpy arrays
    (see _restore_arrays).
    """
    return _read_json(filetoread)


def read_json_dataclass(filetoread, cls=None):
    """Read a JSON file saved with save_json_dataclass and rebuild the dataclass.

    Numeric lists are restored to numpy arrays (see _restore_arrays), then the
    dataclass is reconstructed by keyword: cls(**data). By default the target
    type is cblearn's DimensionEstimationResult, imported lazily so this module
    does not require cblearn unless the default is used. Pass cls=SomeDataclass
    to rebuild a different dataclass.

    Limitation: nested dataclasses are not reconstructed. asdict() flattened
    them to dicts at save time, so they come back as dicts. DimensionEstimation-
    Result has no nested dataclasses (stats_result is a plain dict), so its
    round-trip is exact, with dimensions, train_scores and test_scores restored
    to np.ndarray and estimated_dimension kept as int.
    """
    data = _read_json(filetoread)
    if cls is None:
        from cblearn.embedding import DimensionEstimationResult
        cls = DimensionEstimationResult
    return cls(**data)