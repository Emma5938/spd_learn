# %%
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import pickle
import warnings

from collections import Counter, defaultdict

import numpy as np
import torch


# from juliacall import Main as jl
from torch.utils.data import DataLoader, Dataset, Sampler


###########Handling Julia Package############################
# jl.include(os.path.join(project_root, "utils/precond.jl"))
###########Handling Julia Package############################


def load_covmats(data_dir, db_prefix, file_id=None, verbose=True):
    """Load and concatenate all consistent (cov, label) arrays."""
    if verbose:
        if file_id:
            msg = f"Loading covmats from {db_prefix} database and file ID {file_id}..."
        else:
            msg = f"Loading covmats from {db_prefix} database (all file IDs)..."
        print(msg)
    covs_all, labels_all, dom_all, shapes = [], [], [], []
    candidates = []

    if file_id:  # for a single file (sess-subj)
        prefix = f"{db_prefix}_{file_id}"
        cov_path = os.path.join(data_dir, f"{prefix}_covmats.npy")
        label_path = os.path.join(data_dir, f"{prefix}_labels.npy")
        if os.path.exists(label_path):
            covs = np.load(cov_path)
            candidates.append((f"{prefix}_covmats.npy", covs.shape))
            shapes.append(covs.shape[:2])

    else:  # get all sessions if file_id is not specified
        for fname in os.listdir(data_dir):
            if fname.startswith(db_prefix) and fname.endswith("_covmats.npy"):
                prefix = fname.replace("_covmats.npy", "")
                cov_path = os.path.join(data_dir, fname)
                label_path = os.path.join(data_dir, f"{prefix}_labels.npy")
                if os.path.exists(label_path):
                    covs = np.load(cov_path)
                    candidates.append((fname, covs.shape))
                    shapes.append(covs.shape[:2])

    if not candidates:
        raise ValueError(f"No matching files for prefix '{db_prefix}' in {data_dir}")

    common_shape = Counter(shapes).most_common(1)[0][0]
    mismatched = [f for f, s in candidates if s[:2] != common_shape]
    if mismatched:
        warnings.warn(
            f"Skipping {len(mismatched)} mismatched files: {mismatched}. Most common shape:{common_shape}"
        )

    dom_id = 0
    for fname, shape in candidates:  # apppend only those that match common shape
        if shape[:2] != common_shape:
            continue
        prefix = fname.replace("_covmats.npy", "")
        cov_path = os.path.join(data_dir, fname)
        label_path = os.path.join(data_dir, f"{prefix}_labels.npy")

        covs = np.load(cov_path)
        labels = np.load(label_path)
        covs = np.transpose(covs, (2, 0, 1))  # (n_trials, C, C)
        covs_all.append(covs)
        labels_all.append(labels)
        dom_all.extend(
            [dom_id] * len(labels)
        )  # dom_id for each cov (each file is a diff dom (combination subj-sess))
        dom_id += 1

    covs_all = np.concatenate(covs_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    # convert to right class idx e.g. [1,2] becomes [0,1]
    labels_all = np.array([int(el) - 1 for el in labels_all])
    dom_all = np.array(dom_all)

    if verbose:
        print(
            f"Loaded {len(labels_all)} total trials, shape {common_shape} found at {data_dir}"
        )
    return covs_all, labels_all, dom_all


class CovMatDataset(Dataset):
    def __init__(self, covs, labels, domains):
        self.covs = covs
        self.labels = labels
        self.domains = domains

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cov = torch.from_numpy(self.covs[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        domain = torch.tensor(self.domains[idx]).long()
        return {"data": cov, "label": label, "domain": domain}


class DomainBatchSampler(Sampler):
    def __init__(
        self, domain_ids, batch_size, shuffle=True, drop_last=False, min_batch_size=2
    ):
        """
        Creates batches with unique domains (session or session/subject or session/subject/db)
        with options to drop last batch if batch_size is not reached or more loosely if it doesn't reach min_batch_size
        (domains and samples within domains can be randomized while keeping the unicity of domains in each batch)

        domain_ids: list or array-like of domain id numbers (length = n_epochs)
        batch_size: number of samples per batch
        shuffle: whether to shuffle domains and samples within domains
        """
        self.domain_ids = np.array(domain_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size

        # group sample indices by domain
        self.domain_to_indices = defaultdict(list)
        for idx, dom in enumerate(self.domain_ids):
            self.domain_to_indices[dom].append(idx)

        self.domains = list(self.domain_to_indices.keys())

    def __iter__(self):
        domains = self.domains.copy()
        if self.shuffle:
            np.random.shuffle(domains)  # shuffle domain instead of all data samples

        all_batches = []

        for dom in domains:
            indices = self.domain_to_indices[dom]
            if self.shuffle:
                np.random.shuffle(indices)  # shuffle data samples inside each domain
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last:  # drop last if length is inferior to batch_size
                    if len(batch) == self.batch_size:
                        all_batches.append(batch)
                else:
                    if (
                        len(batch) >= self.min_batch_size
                    ):  # drop batch if length is inferior to min_batch_size
                        all_batches.append(batch)

        if self.shuffle:
            np.random.shuffle(
                all_batches
            )  # shuffle batches order (domains/sessions might appear non-contiguously)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.domain_to_indices.values():
            n = len(indices)
            n_batches = n // self.batch_size
            if not self.drop_last and n % self.batch_size >= self.min_batch_size:
                n_batches += 1
            total_batches += n_batches
        return total_batches


def get_eeg_loaders_dep(
    db_prefix,
    data_dir,
    batch_size,
    precond=False,
    precond_explVar=1,
    batch_single_dom=True,
    min_batch_size=5,
    file_id=None,
    train_pct=0.7,
    test_pct=0.15,
):
    """
    Deprecated.
    Get arrays of cov matrices and labels, split, (precondition) and return train, test, val dataloaders.
    Does not make a distinction of domains neither in train,test,val splits neither in preconditioning
    (we do not even guarantee we see all domains in train split).
    db_prefix: string with the name of the database used in the saved files
    data_dir: str path to the files
    train_test_split_rate
    precond_explVar: float in ]0,1] . Preconditioning explained variance. If equals 1, no dimensionality reduction will be performed.
    """
    # load
    covs, labels, domains = load_covmats(
        data_dir=data_dir, db_prefix=db_prefix, file_id=file_id
    )
    # split
    assert (
        train_pct + test_pct <= 1.0
    ), "Train and test split rate should sum less then 1."
    n = len(labels)
    n_train, n_test = int(train_pct * n), int(test_pct * n)
    n_val = n - n_train - n_test
    train_idx, val_idx, test_idx = torch.utils.data.random_split(
        range(n), [n_train, n_val, n_test]
    )
    # here we do not guarantee that all domains will be covered during training.
    # see next function for that.

    # Precondition if asked (to all domains at the same time)
    if precond:
        print("Applying preconditioning...")
        if precond_explVar == 1:
            eVar = covs.shape[-1]  # n = last dimension of covs
        else:
            eVar = precond_explVar
        # create the pipeline via the module
        pipeline = jl.Precond.make_pipeline(eVar)
        # call the Julia function pre_cond with numpy arrays
        C_train_new, C_test_new, C_val_new = jl.Precond.pre_cond(
            covs[train_idx.indices],
            covs[test_idx.indices],
            covs[val_idx.indices],
            pipeline,
        )

        # Convert the returned Julia arrays back to numpy
        covs[train_idx.indices] = np.array(C_train_new)
        covs[test_idx.indices] = np.array(C_test_new)
        covs[val_idx.indices] = np.array(C_val_new)
        print(
            "Train Covariance matrices were preconditioned. ",
            f"Precondition was also applied to test and val. New shape: {covs.shape[-2:]}",
        )

    train_set = CovMatDataset(
        covs[train_idx.indices], labels[train_idx.indices], domains[train_idx.indices]
    )
    val_set = CovMatDataset(
        covs[val_idx.indices], labels[val_idx.indices], domains[val_idx.indices]
    )
    test_set = CovMatDataset(
        covs[test_idx.indices], labels[test_idx.indices], domains[test_idx.indices]
    )

    # Get batches containing a single domain if asked
    if batch_single_dom:
        # samplers for iterating over domains/sessions respecting min_batch_size
        sampler_train = DomainBatchSampler(
            domains[train_idx.indices],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,  # keep small batches
            min_batch_size=min_batch_size,  # avoid batches with less than min_batch_size
        )
        sampler_val = DomainBatchSampler(
            domains[val_idx.indices],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,  # keep small batches
            min_batch_size=min_batch_size,  # avoid batches with less than min_batch_size
        )
        sampler_test = DomainBatchSampler(
            domains[test_idx.indices],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,  # keep small batches
            min_batch_size=min_batch_size,  # avoid batches with less than min_batch_size
        )
        train_loader = DataLoader(train_set, batch_sampler=sampler_train)
        val_loader = DataLoader(val_set, batch_sampler=sampler_val)
        test_loader = DataLoader(test_set, batch_sampler=sampler_test)

    else:  # if we dont care if each batch contains a single domain
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader


def get_arrays_treat_by_domain(
    db_prefix,
    data_dir,
    precond=False,
    precond_explVar=1,
    file_id=None,
    train_pct=0.7,
    test_pct=0.15,
):
    covs, labels, domains = load_covmats(
        data_dir=data_dir, db_prefix=db_prefix, file_id=file_id
    )
    assert train_pct + test_pct <= 1.0, "Train and test split rates must sum to <= 1."

    rng = np.random.default_rng()

    splits = {"train": [], "val": [], "test": []}

    for dom in np.unique(domains):
        mask = domains == dom
        covs_dom, labels_dom = covs[mask], labels[mask]
        n = len(labels_dom)

        n_train = int(train_pct * n)
        n_test = int(test_pct * n)
        n_val = n - n_train - n_test

        idx = rng.permutation(n)
        i_train, i_val, i_test = (
            idx[:n_train],
            idx[n_train : n_train + n_val],
            idx[n_train + n_val :],
        )

        # precondition if asked
        if precond:
            print(f"Applying preconditioning to domain {dom}...")
            eVar = covs.shape[-1] if precond_explVar == 1 else precond_explVar
            pipeline = jl.Precond.make_pipeline(eVar)
            C_train, C_test, C_val = jl.Precond.pre_cond(
                covs_dom[i_train], covs_dom[i_test], covs_dom[i_val], pipeline
            )
            covs_dom[i_train], covs_dom[i_test], covs_dom[i_val] = map(
                np.array, (C_train, C_test, C_val)
            )
            print(
                f"Preconditioning done for domain {dom}: new shape {covs_dom.shape[-2:]}"
            )

        # collect all domain-specific splits
        splits["train"].append(
            (covs_dom[i_train], labels_dom[i_train], np.full(len(i_train), dom))
        )
        splits["val"].append(
            (covs_dom[i_val], labels_dom[i_val], np.full(len(i_val), dom))
        )
        splits["test"].append(
            (covs_dom[i_test], labels_dom[i_test], np.full(len(i_test), dom))
        )

    # concatenate all domains for each split
    def cat_split(key):
        covs_all, labels_all, doms_all = zip(*splits[key])
        return (
            np.concatenate(covs_all),
            np.concatenate(labels_all),
            np.concatenate(doms_all),
        )

    covs_train, labels_train, doms_train = cat_split("train")
    covs_val, labels_val, doms_val = cat_split("val")
    covs_test, labels_test, doms_test = cat_split("test")

    return (
        (covs_train, labels_train, doms_train),
        (covs_val, labels_val, doms_val),
        (covs_test, labels_test, doms_test),
    )


def get_eeg_loaders(
    *,
    processed_root,
    db_prefix,
    data_dir,
    batch_size,
    batch_single_dom,
    precond,
    precond_explVar,
    file_id,
    train_pct,
    test_pct,
    k_folds,
    seed,
    fold_id,
    min_batch_size=5,
    dom_new_label=None,
):
    """
    Load preprocessed EEG data from disk and build loaders.
    """

    from utils.data_helpers import make_data_signature

    signature = make_data_signature(
        db_prefix=db_prefix,
        data_dir=data_dir,
        file_id=file_id,
        precond=precond,
        precond_explVar=precond_explVar,
        batch_single_dom=batch_single_dom,
        train_pct=train_pct,
        test_pct=test_pct,
        k_folds=k_folds,
        seed=seed,
    )

    fold_path = os.path.join(processed_root, signature, f"fold_{fold_id}.pkl")
    if not os.path.exists(fold_path):
        raise FileNotFoundError(
            f"Preprocessed fold not found: {fold_path}\n"
            f"Run preprocess_eeg_data first. Run \n \
            $ python -m data_scripts.preprocess_data --config configs/config.yaml"
        )

    with open(fold_path, "rb") as f:
        data = pickle.load(f)

    def build(split):
        covs, labels, doms = data[split]
        if dom_new_label is not None:
            doms = np.full_like(doms, dom_new_label)
        return CovMatDataset(covs, labels, doms), doms

    train_set, doms_train = build("train")
    val_set, doms_val = build("val")
    test_set, doms_test = build("test")

    if batch_single_dom:

        def make_sampler(dom_array, shuffle):
            return DomainBatchSampler(
                dom_array,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                min_batch_size=min_batch_size,
            )

        train_loader = DataLoader(
            train_set, batch_sampler=make_sampler(doms_train, True)
        )
        val_loader = DataLoader(val_set, batch_sampler=make_sampler(doms_val, False))
        test_loader = DataLoader(test_set, batch_sampler=make_sampler(doms_test, False))
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


# def get_eeg_loaders(
#     db_prefix,
#     data_dir,
#     batch_size,
#     precond=False,
#     precond_explVar=1,
#     batch_single_dom=True,
#     min_batch_size=5,
#     file_id=None,
#     train_pct=0.7,
#     test_pct=0.15,
#     dom_new_label = None
# ):
#     """
#     Load, split, (optionally precondition per domain), and return EEG covariance DataLoaders per domain.
#     """

#     (covs_train, labels_train, doms_train), (covs_val, labels_val, doms_val),\
#           (covs_test, labels_test, doms_test) = get_arrays_treat_by_domain(db_prefix,
#                                                                             data_dir,
#                                                                             precond=precond,
#                                                                             precond_explVar=precond_explVar,
#                                                                             file_id=file_id,
#                                                                             train_pct=train_pct,
#                                                                             test_pct=test_pct)
#     if dom_new_label is not None:
#         #for multidatabase setting. Domain label will be the database name index (preprocessing is still the same)
#         doms_train = np.full(doms_train.shape, dom_new_label)
#         doms_val = np.full(doms_val.shape, dom_new_label)
#         doms_test = np.full(doms_test.shape, dom_new_label)
#         print(doms_train)

#     #build datasets
#     train_set = CovMatDataset(covs_train, labels_train, doms_train)
#     val_set = CovMatDataset(covs_val, labels_val, doms_val)
#     test_set = CovMatDataset(covs_test, labels_test, doms_test)

#     #build loaders. We can still precondition per domain and have a domain unaware split
#     if batch_single_dom:
#         def make_sampler(dom_array, shuffle):
#             return DomainBatchSampler(
#                 dom_array,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 drop_last=False,
#                 min_batch_size=min_batch_size,
#             )

#         train_loader = DataLoader(train_set, batch_sampler=make_sampler(doms_train, True))
#         val_loader = DataLoader(val_set, batch_sampler=make_sampler(doms_val, False))
#         test_loader = DataLoader(test_set, batch_sampler=make_sampler(doms_test, False))
#     else:
#         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader, val_loader


if __name__ == "__main__":
    # directory containing covs files (this is for MI, do later for P300)
    data_dir = "/localdata/costamai/Apps/LibData/NY/MI/ExtractedCovMats_fromscript"
    db_prefix = "BNCI2015001"

    t, te, v = get_eeg_loaders_dep(
        db_prefix,
        data_dir,
        batch_size=40,
        precond=False,
        batch_single_dom=False,
        min_batch_size=5,
        file_id=None,
        train_pct=0.7,
        test_pct=0.15,
    )

    # load_covmats(data_dir=data_dir, db_prefix=db_prefix)

    # batch_size = 60
    # train_loader, test_loader, val_loader = get_eeg_loaders_dep(db_prefix = db_prefix,
    #                                                                        data_dir=data_dir,
    #                                                                        batch_size=batch_size,
    #                                                                        file_id=1)
# %%