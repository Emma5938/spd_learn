import datetime
import hashlib
import json
import os
import platform
import sys

from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Expand environment variables in data_dir / processed_root for every
    # dataset sub-section that has them (eeg, loso, sim).
    #
    # IMPORTANT — hash stability: make_data_signature hashes data_dir as a
    # string. The original code used Path(...).expanduser().resolve(), so
    # existing processed data on disk was hashed with the resolved absolute
    # path string. We must do the same here, otherwise the hash changes and
    # the preprocessed fold files can no longer be found.
    #
    # Consequence: data_dir in your config must expand/resolve to the SAME
    # string that was used when you ran preprocess_data originally. If you
    # used a literal path before, use the same literal path (or an env var
    # that expands to exactly that string) in the new config.
    for section in ("eeg", "loso", "sim"):
        sub = cfg.get("data", {}).get(section)
        if sub is None:
            continue
        for key in ("data_dir", "processed_root"):
            if key not in sub:
                continue
            val = os.path.expandvars(str(sub[key]))
            if "$" in val:
                raise RuntimeError(
                    f"Environment variable not set in data.{section}.{key}: {sub[key]}"
                )
            sub[key] = Path(val).expanduser().resolve()

    return cfg


def make_data_signature(**kwargs):
    """
    Create a stable hash from all data-related arguments.

    Normalisation rules (must match what was applied when the data was
    originally preprocessed):
      - Path objects and strings are both converted to str via default=str
      - This is identical to the original behaviour in get_eeg_data.py
    """
    payload = json.dumps(
        kwargs,
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _compute_fold_statistics(folds):
    """
    folds: dict[int, dict[str, list[np.ndarray]]]
           folds[fold_id][split] = list of index arrays (per domain)

    Returns a clean, serializable summary.
    """
    stats = {}

    for fold_id, fold_data in folds.items():
        stats[f"fold_{fold_id}"] = {}

        for split, indices_list in fold_data.items():
            # concatenate domain-wise indices
            all_indices = np.concatenate(indices_list) if indices_list else []
            stats[f"fold_{fold_id}"][split] = {
                "n_samples": int(len(all_indices)),
            }

    return stats


def _yaml_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    else:
        return obj


def write_metadata(
    path,
    *,
    data_cfg,
    experiment_cfg,
    folds,
    extra=None,
):
    """
    Write a metadata.yaml file describing how the data was generated.
    """

    metadata = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "data_specification": _yaml_safe(data_cfg),
        "experiment_specification": _yaml_safe(experiment_cfg),
        "fold_statistics": _compute_fold_statistics(folds),
    }

    if extra is not None:
        metadata["extra"] = _yaml_safe(extra)

    with open(path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

def get_class_name(cls):
    if cls==0:
        return "right hand"
    if cls==1:
        return "feet"