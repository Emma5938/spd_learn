#%%
"""
Generate k-fold preprocessed data using julia script and saves in processed_root folder.
Data configs are defined in the config file.

Usage:

$ python -m data_scripts.preprocess_data --config configs/config.yaml

"""

import os

#to avoid julia core dumped
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JULIA_NUM_THREADS"] = "1"

import pickle
import sys
import argparse

from collections import defaultdict

import numpy as np

from juliacall import Main as jl
# Installe le package uniquement dans l'environnement caché de juliacall
jl.seval('import Pkg; Pkg.add("PosDefManifold"); Pkg.add("PosDefManifoldML"); Pkg.add("LinearAlgebra")')
from sklearn.model_selection import KFold


# project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))               # chemin vers eeg_data
sys.path.append(project_root)

from data_scripts.get_eeg_data import load_covmats
from utils.data_helpers import make_data_signature, write_metadata, load_config


# load Julia module once
jl.include(os.path.join(project_root, "utils/precond.jl"))


def preprocess_eeg_data(
    *,
    db_prefix,
    data_dir,
    file_id,
    precond, # if true apply pre-conditioning
    precond_tikhonov : float = 1e-8, # Tikhonov regularization 
    precond_explVar : int = 0, # if zero automatically reduce the dimension, otherwise see eVar in PosDefManifoldML.jl
    batch_single_dom,
    train_pct,
    test_pct,
    k_folds,
    seed,
    output_root,
    save_iZ: bool = False, 
):
    """
    Generate K-fold preprocessed EEG data and save to disk.
    Preconditioning is applied per domain immediately after K-fold split.

    For each domain d, generates K-Folds (train_dk, test_dk,  val_dk) and applies pre-condiitoning (per domain).
    Returns:
            output_dir
    When save_iZ=True, the back-projection matrix iZ returned by Julia's
    preconditioning is stored inside each fold pkl under the key "iZ" as
    {domain_id (int): np.ndarray (n_chans, n_chans)}. Also changes the
    signature hash so this data lives in a different directory from old data.

    Saves to disk K pickle files:
        fold_k.pkl -> {"train": (C,y,d), "val": ..., "test": ...,
                       "iZ": {dom: array}}  # iZ only when save_iZ=True
    """

    # ---------- data signature ----------
    signature = make_data_signature(
        db_prefix=db_prefix,
        data_dir=data_dir,
        file_id=file_id,
        precond=precond,
        precond_tikhonov = precond_tikhonov, 
        precond_explVar=precond_explVar,
        batch_single_dom=batch_single_dom,
        train_pct=train_pct,
        test_pct=test_pct,
        k_folds=k_folds,
        seed=seed,
        # Only included when True so old data (no key) still resolves correctly
        **( {"save_iZ": True} if save_iZ else {} ),
    )

    out_dir = os.path.join(output_root, signature)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- load raw data ----------
    covs, labels, domains = load_covmats(
        data_dir=data_dir,
        db_prefix=db_prefix,
        file_id=file_id,
    )

    rng = np.random.default_rng(seed)

    # ---------- domain-aware K-fold ----------
    domain_indices = {dom: np.where(domains == dom)[0] for dom in np.unique(domains)}

    # folds: indices ONLY (for metadata)
    folds = defaultdict(lambda: {"train": [], "val": [], "test": []})

    # folds_data: actual (C, y, d) tuples
    folds_data = defaultdict(lambda: {"train": [], "val": [], "test": []})
    # iZ_data[fold_id][domain_id] = np.ndarray (n_chans, n_chans)
    iZ_data: dict = defaultdict(dict)

    for dom, idx in domain_indices.items():
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(idx)):
            idx_trainval = idx[trainval_idx]
            idx_test = idx[test_idx]

            # secondary split train / val
            n_train = int(train_pct * len(idx_trainval))
            perm = rng.permutation(len(idx_trainval))
            train_idx = idx_trainval[perm[:n_train]]
            val_idx = idx_trainval[perm[n_train:]]

            # ---- store indices for metadata (UNCHANGED BEHAVIOR)
            folds[fold_id]["train"].append(train_idx)
            folds[fold_id]["val"].append(val_idx)
            folds[fold_id]["test"].append(idx_test)

            # ---------- Julia preconditioning per domain ----------
            if precond:
                # eVar = covs.shape[-1] if precond_explVar == 1 else precond_explVar

                C_train_dom = covs[train_idx]
                C_val_dom   = covs[val_idx]
                C_test_dom  = covs[idx_test]
                print("precond_explVar=", precond_explVar)
                print("precond_tikhonov=", precond_tikhonov)
                print("covs.shape[-1]=", covs.shape[-1])
                # n = C_train_dom.shape[1]  # normalement 14 dans ton cas
                pipeline = jl.Precond.make_pipeline(precond_tikhonov, precond_explVar, covs.shape[-1] )                                    # eVar n'est pas utilisé  # make_pipeline(n)

                print("pipeline = ", pipeline)

                C_train_p, C_test_p, C_val_p= jl.Precond.pre_cond(                                      # Enlever iZ avec C_val_p et ajout de pipeline après C_val_dom 
                    C_train_dom, C_test_dom, C_val_dom, pipeline                 
                )
                # iZ: (n_chans, n_chans) — back-projection matrix such that
                # iZ.T @ (precond_cov / beta) @ iZ ≈ original sensor-space cov

                print(f"C_train shape: {np.array(C_train_p).shape}")
                print(f"C_test  shape: {np.array(C_test_p).shape}")
                print(f"C_val   shape: {np.array(C_val_p).shape}")
                # print(f"iZ      shape: {np.array(iZ).shape}")

                folds_data[fold_id]["train"].append(
                    (np.array(C_train_p), labels[train_idx], domains[train_idx])
                )
                folds_data[fold_id]["val"].append(
                    (np.array(C_val_p), labels[val_idx], domains[val_idx])
                )
                folds_data[fold_id]["test"].append(
                    (np.array(C_test_p), labels[idx_test], domains[idx_test])
                )

                if save_iZ:
                    iZ_data[fold_id][int(dom)] = np.array(iZ)
            else:
                folds_data[fold_id]["train"].append(
                    (covs[train_idx], labels[train_idx], domains[train_idx])
                )
                folds_data[fold_id]["val"].append(
                    (covs[val_idx], labels[val_idx], domains[val_idx])
                )
                folds_data[fold_id]["test"].append(
                    (covs[idx_test], labels[idx_test], domains[idx_test])
                )

    # ---------- write metadata ----------
    metadata_path = os.path.join(out_dir, "metadata.yaml")

    if not os.path.exists(metadata_path):
        write_metadata(
            metadata_path,
            data_cfg=dict(
                db_prefix=db_prefix,
                data_dir=data_dir,
                file_id=file_id,
                precond=precond,
                precond_tikhonov = precond_tikhonov,
                precond_explVar=precond_explVar,
                batch_single_dom=batch_single_dom,
                train_pct=train_pct,
                test_pct=test_pct,
                k_folds=k_folds,
                seed=seed,
                save_iZ=save_iZ,
            ),
            experiment_cfg=dict(
                signature=signature,
                output_root=output_root,
            ),
            folds=folds,  # ✅ indices only
            extra=dict(
                julia_pipeline=(
                    f"Tikhonov(precond_tikhonov) → Recenter(eVar={precond_explVar}) → Equalize"
                    if precond
                    else None
                ),
            ),
        )

    # ---------- assemble and save fold pkls ----------
    for fold_id in range(k_folds):
        fold_data = {}

        for split in ("train", "val", "test"):
            C_list, y_list, d_list = zip(*folds_data[fold_id][split])
            fold_data[split] = (
                np.concatenate(C_list, axis=0),
                np.concatenate(y_list, axis=0),
                np.concatenate(d_list, axis=0),
            )

        # iZ dict: {domain_id: (n_chans, n_chans)} — only when save_iZ=True
        if save_iZ and iZ_data[fold_id]:
            fold_data["iZ"] = iZ_data[fold_id]

        pkl_path = os.path.join(out_dir, f"fold_{fold_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(fold_data, f)
        print(f"Saved {pkl_path}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data (K-fold, domain-aware, cached)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="processed_data",
        help="Root directory for processed datasets",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # -------- extract sections --------
    data_cfg = cfg["data"]["eeg"]
    exp_cfg = cfg["experiment"]

    # -------- call preprocessing --------
    preprocess_eeg_data(
        db_prefix=data_cfg["db_prefix"],
        data_dir=data_cfg["data_dir"],
        file_id=data_cfg.get("file_id"),
        precond=data_cfg["precond"],
        precond_explVar=data_cfg["precond_explVar"],
        batch_single_dom=data_cfg["batch_single_dom"],
        train_pct=data_cfg.get("train_pct", 0.7),
        test_pct=data_cfg.get("test_pct", 0.15),
        k_folds=data_cfg["k_folds"],
        seed=exp_cfg["seed"],
        output_root=data_cfg["processed_root"],
        save_iZ=data_cfg.get("save_iZ", False),
    )



if __name__ == "__main__":
    main()

# %%