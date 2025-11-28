# generate_splits.py
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges,from_scipy_sparse_matrix
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import List, Tuple

def generate_link_splits(adj_csr: sp.csr_matrix, test_ratio: float = 0.2, n_splits: int = 5, seed: int = 42, save_dir: str = "splits"):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    seeds = [seed + i for i in range(n_splits)]
    splits = []

    edge_index, _ = from_scipy_sparse_matrix(adj_csr)
    edge_index = edge_index.cpu()
    edges = edge_index.cpu().numpy().T  # (E, 2)
    E = len(edges)
    n_nodes = adj_csr.shape[0]

    for i, s in enumerate(seeds):
        np.random.seed(s)

        mask = np.zeros((n_nodes, n_nodes), dtype=bool)
        mask[edges[:, 0], edges[:, 1]] = True
        mask[edges[:, 1], edges[:, 0]] = True
        neg_candidates = np.where(~mask)
        neg_idx = np.random.choice(len(neg_candidates[0]), E, replace=False)
        neg_all = np.vstack([neg_candidates[0][neg_idx], neg_candidates[1][neg_idx]]).T

        pos_labels = np.ones(E)
        neg_labels = np.zeros(E)
        all_edges = np.vstack([edges, neg_all])
        all_labels = np.hstack([pos_labels, neg_labels])

        train_idx, test_idx = train_test_split(
            np.arange(2*E), test_size=test_ratio, random_state=s, stratify=all_labels
        )

        train_edges = all_edges[train_idx]
        test_edges = all_edges[test_idx]
        train_labels = all_labels[train_idx]
        test_labels = all_labels[test_idx]

        # 4. 分离正负
        pos_train = train_edges[train_labels == 1]
        neg_train = train_edges[train_labels == 0]
        pos_test = test_edges[test_labels == 1]
        neg_test = test_edges[test_labels == 0]

        # 5. 训练图
        adj_train = sp.csr_matrix(
            (np.ones(len(pos_train)), (pos_train[:, 0], pos_train[:, 1])),
            shape=adj_csr.shape
        ).tocsr()

        split = {
            'adj_train': adj_train,
            'pos_train': train_edges,
            'neg_train': neg_train,
            'pos_test': pos_test,
            'neg_test': neg_test,
        }
        splits.append(split)

        # 保存
        path = os.path.join(save_dir, f"link_{i}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(split, f)

    return splits

def generate_node_splits(labels: np.ndarray, test_size: float = 0.2, n_splits: int = 5, seed: int = 42, save_dir: str = "splits"):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    seeds = [seed + i for i in range(n_splits)]
    splits = []

    for i, s in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            np.arange(len(labels)), labels,
            test_size=test_size, random_state=s, stratify=labels
        )
        split = {
            'train_idx': X_train,
            'test_idx': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }
        splits.append(split)

        path = os.path.join(save_dir, f"node_{i}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(split, f)

    return splits

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name")
    args = parser.parse_args()

    edgelist = np.loadtxt(f'data/{args.dataset}/edgelist.txt', dtype=int)
    dataset_to_nnodes = {'cora': 2708,'citeseer': 3312,'wiki': 2405,'blogcatalog': 5196}
    n_nodes = dataset_to_nnodes[args.dataset]
    adj = sp.coo_matrix((np.ones(edgelist.shape[0]), (edgelist[:, 0], edgelist[:, 1])), shape=(n_nodes, n_nodes))
    adj = adj + adj.T  # Make it undirected
    adj[adj > 1] = 1  # Remove multi-edges
    adj = adj.tocsr()
    n_nodes = adj.shape[0]
    m_edges = adj.nnz
    labels = np.loadtxt(f'data/{args.dataset}/labels.txt', dtype=int)[:, 1]
    print(labels.shape)
    print(f"Dataset: {args.dataset}, Nodes: {n_nodes}, Edges: {m_edges}")
    sp.save_npz(f"./data/{args.dataset}/adj.npz",adj) 
    np.save(f"./data/{args.dataset}/labels.npy", labels)
    generate_link_splits(adj, save_dir=f"splits/{args.dataset}_link", n_splits=5)
    generate_node_splits(labels, save_dir=f"splits/{args.dataset}_node", n_splits=5)