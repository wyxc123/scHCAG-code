

import numpy as np
from scipy import sparse as sp
import scanpy as sc
import anndata

from .graph_function import get_appro_adj

DEFAULT_MIN_RESOLUTION = 0.1
DEFAULT_MAX_RESOLUTION = 2.0
DEFAULT_RESOLUTION_STEP = 0.05


def make_undirected_adjacency(adj, remove_self_loop=True):
   
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = adj.tocsr()

    
    adj = adj.maximum(adj.T)

    if remove_self_loop:
        adj.setdiag(0)
        adj.eliminate_zeros()

    return adj.tocsr()


def compute_graph_isi(adj, labels, eps=1e-12):
    
    adj = make_undirected_adjacency(adj)

    labels = np.asarray(labels)
    N = adj.shape[0]

    if labels.shape[0] != N:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match adjacency size {N}."
        )

    
    unique_labels, labels_encoded = np.unique(labels, return_inverse=True)
    m = len(unique_labels)

    if m <= 1 or m >= N:
        return 0.0

   
    one_hot = sp.csr_matrix(
        (np.ones(N), (np.arange(N), labels_encoded)),
        shape=(N, m)
    )

    
    internal_mat = one_hot.T @ adj @ one_hot
    internal = np.asarray(internal_mat.diagonal()).flatten() / 2.0

    degree = np.asarray(adj.sum(axis=1)).flatten()
    total_deg = np.asarray(one_hot.T @ degree).flatten()

    
    external = total_deg - 2.0 * internal
    external = np.maximum(external, 0.0)

    sizes = np.asarray(one_hot.sum(axis=0)).flatten()

    Coh = np.zeros(m, dtype=np.float64)
    Sep = np.zeros(m, dtype=np.float64)

    for i in range(m):
        size = sizes[i]

       
        max_internal = size * (size - 1.0) / 2.0
        if max_internal > 0:
            Coh[i] = internal[i] / (max_internal + eps)
        else:
            Coh[i] = 0.0

        
        max_external = size * (N - size)
        if max_external > 0:
            Sep[i] = external[i] / (max_external + eps)
        else:
            Sep[i] = 0.0

    
    Q = Coh / (Coh + Sep + eps)

    
    Q_graph = float(np.mean(Q))

    
    p = sizes / float(N)

    
    entropy = -np.sum(p * np.log(p + eps))
    H_norm = entropy / (np.log(m + eps) + eps)

    
    CV = np.std(p) / (np.mean(p) + eps)
    CV_n = CV / (1.0 + CV)

    
    entropy_penalty = np.exp(
        -H_norm * (np.log(m + eps) / (np.log(N + eps) + eps))
    )
    cv_penalty = np.exp(-CV_n)

    isi = Q_graph * entropy_penalty * cv_penalty

    return float(isi)


def run_gisi(
    feature,
    pre_adj=None,
    k_neighbors=15,
    resolutions=None,
    leiden=True,
    louvain=False,
    log=None,
    use_feature_graph=True
):
    
    feature = np.asarray(feature)

    if feature.ndim != 2:
        raise ValueError("feature must be a 2D array with shape (n_samples, n_features).")

    n_samples = feature.shape[0]





    if resolutions is None:
        resolutions = np.arange(
            DEFAULT_MIN_RESOLUTION,
            DEFAULT_MAX_RESOLUTION + DEFAULT_RESOLUTION_STEP / 2,
            DEFAULT_RESOLUTION_STEP
        )

        if resolutions[-1] < DEFAULT_MAX_RESOLUTION:
            resolutions = np.append(
                resolutions,
                DEFAULT_MAX_RESOLUTION
            )

        resolutions = np.unique(
            np.round(resolutions, decimals=10)
        )

    if not leiden and not louvain:
        leiden = True
        if log is not None:
            log.info("No graph clustering method specified; defaulting to Leiden.")

    
    if use_feature_graph:
        if log is not None:
            log.info(
                f"Building KNN graph from current embedding "
                f"(n={n_samples}, k={k_neighbors})."
            )
        adj = get_appro_adj(feature, k=k_neighbors)
    else:
        if pre_adj is not None and pre_adj.shape[0] == n_samples:
            if log is not None:
                log.info("Using precomputed adjacency matrix for GISI.")
            adj = pre_adj
        else:
            if log is not None:
                log.warning(
                    "pre_adj is None or shape-mismatched; "
                    "building KNN graph from current embedding instead."
                )
            adj = get_appro_adj(feature, k=k_neighbors)

    
    adj = make_undirected_adjacency(adj)

    N = adj.shape[0]

    adata = anndata.AnnData(feature)
    adata.obsp["connectivities"] = adj
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities"
    }

    best_isi = -1.0
    best_labels = None
    best_res = None

    for res in resolutions:
        if leiden:
            sc.tl.leiden(
                adata,
                resolution=float(res),
                key_added="cluster"
            )
        elif louvain:
            sc.tl.louvain(
                adata,
                resolution=float(res),
                key_added="cluster"
            )

        labels = adata.obs["cluster"].cat.codes.values
        m = len(np.unique(labels))

        if m <= 1 or m >= N:
            isi = 0.0
        else:
            isi = compute_graph_isi(adj, labels)

        if log is not None:
            log.info(
                f"Resolution {res:.2f}, clusters: {m}, "
                f"Graph-ISI*: {isi:.6f}"
            )

        if isi > best_isi:
            best_isi = isi
            best_labels = labels.copy()
            best_res = float(res)

    if best_labels is None:
        raise RuntimeError("GISI failed to find a valid clustering result.")

    return best_labels, best_res, best_isi
