# embedding_framework_extended.py
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from typing import Dict, Any, Tuple, List
import warnings
import time
warnings.filterwarnings("ignore")

# For advanced methods
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import from_scipy_sparse_matrix, negative_sampling, train_test_split_edges
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from torch_geometric.nn.models import Node2Vec
from torch_geometric.data import Data
from gensim.models import Word2Vec

class GraphEmbeddingFramework:
    def __init__(self, adj_csr: sparse.csr_matrix, embedding_dim: int = 128, 
                 walk_length: int = 80, num_walks: int = 10, window_size: int = 10,
                 workers: int = 4, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.adj_csr = adj_csr
        self.graph = nx.from_scipy_sparse_array(adj_csr)
        self.num_nodes = adj_csr.shape[0]
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        self.device = torch.device(device)

        self.embeddings = None
        self.edge_index = from_scipy_sparse_matrix(adj_csr)[0].to(self.device)

    # ========================== EMBEDDING METHODS ==========================
    
    def fit_deepwalk(self):
        print("DeepWalk: Generating walks...")
        walks = self._generate_walks()
        str_walks = [[str(n) for n in walk] for walk in walks]
        model = Word2Vec(str_walks, vector_size=self.embedding_dim, window=self.window_size,
                         min_count=0, sg=1, workers=self.workers, epochs=5)
        self.embeddings = np.array([model.wv[str(i)] for i in range(self.num_nodes)])
        print(f"DeepWalk embeddings: {self.embeddings.shape}")
        return self.embeddings

    def _generate_walks(self):
        nodes = list(range(self.num_nodes))
        walks = []
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(node)
                walks.append(walk)
        return walks

    def _random_walk(self, start_node: int):
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = self.adj_csr.indices[self.adj_csr.indptr[current]:self.adj_csr.indptr[current + 1]]
            if len(neighbors) == 0:
                break
            walk.append(random.choice(neighbors))
        return walk

    def fit_node2vec(self, p: float = 1.0, q: float = 1.0, epochs: int = 10):
        print("Node2Vec: Training...")
        start_time = time.time()
        model = Node2Vec(
            self.edge_index, embedding_dim=self.embedding_dim,
            walk_length=self.walk_length, context_size=self.window_size,
            walks_per_node=self.num_walks, p=p, q=q,
            num_negative_samples=1, sparse=True
        ).to(self.device)
        model.to(self.device)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # if epoch % 2 == 0:
            #     print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
        end_time = time.time()
        print(f"Node2Vec training completed in {end_time - start_time:.2f} s.")
        self.embeddings = model.embedding.weight.cpu().detach().numpy()
        print(f"Node2Vec embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_netmf(self):
        print("NetMF: Computing...")
        vol = self.adj_csr.sum()
        deg = np.array(self.adj_csr.sum(axis=1)).squeeze()
        walks = self._generate_walks()
        
        cooccur = np.zeros((self.num_nodes, self.num_nodes))
        for walk in walks:
            for i in range(len(walk)):
                for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
                    if i != j:
                        cooccur[walk[i], walk[j]] += 1 / abs(i - j)
        
        log_co = np.log(cooccur + 1)
        log_deg = np.log(deg + 1)
        M = vol * (log_co - np.outer(log_deg, np.ones(self.num_nodes)))
        M = np.maximum(M, 0)
        
        from scipy.sparse.linalg import svds
        U, S, _ = svds(M, k=min(self.embedding_dim, M.shape[0]-1))
        self.embeddings = U @ np.diag(np.sqrt(S))
        print(f"NetMF embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_sdne(self, epochs: int = 50):
        print("SDNE: Training...")
        class SDNE(nn.Module):
            def __init__(self, n_nodes, dim):
                super().__init__()
                self.enc1 = nn.Linear(n_nodes, 512)
                self.enc2 = nn.Linear(512, dim)
                self.dec2 = nn.Linear(dim, 512)
                self.dec1 = nn.Linear(512, n_nodes)
            
            def forward(self, x):
                h = torch.sigmoid(self.enc1(x))
                z = self.enc2(h)
                h = torch.sigmoid(self.dec2(z))
                x_hat = self.dec1(h)
                return z, x_hat

        adj_norm = self.adj_csr / (self.adj_csr.sum(axis=1).A1 + 1e-8)
        adj_t = torch.FloatTensor(adj_norm.toarray()).to(self.device)
        x = torch.eye(self.num_nodes).to(self.device)

        model = SDNE(self.num_nodes, self.embedding_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            z, x_hat = model(x)
            loss = torch.mean((x_hat - adj_t) ** 2) + 1e-5 * torch.mean(z ** 2)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        with torch.no_grad():
            self.embeddings = model(x)[0].cpu().numpy()
        print(f"SDNE embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_gcn(self, epochs: int = 100):
        print("GCN: Training (unsupervised link prediction)...")
        from torch_geometric.nn import GCNConv

        class GCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        x = torch.eye(self.num_nodes).to(self.device)

        model = GCN(
            in_channels=self.num_nodes,
            hidden_channels=256,
            out_channels=self.embedding_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = model(x, self.edge_index)

            pos_score = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=-1)

            neg_edge_index = negative_sampling(self.edge_index, num_nodes=self.num_nodes)
            neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

            loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() \
                   - torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:03d}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            self.embeddings = model(x, self.edge_index).cpu().numpy()

        print(f"GCN embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_gin(self, epochs: int = 100):
        print("GIN: Training (unsupervised link prediction)...")
        from torch_geometric.nn import GINConv

        class GIN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                # GIN 需要手动构造 MLP
                self.mlp1 = nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.mlp2 = nn.Sequential(
                    nn.Linear(hidden_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
                self.conv1 = GINConv(self.mlp1)
                self.conv2 = GINConv(self.mlp2)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        x = torch.eye(self.num_nodes).to(self.device)

        model = GIN(
            in_channels=self.num_nodes,
            hidden_channels=256,
            out_channels=self.embedding_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = model(x, self.edge_index)

            pos_score = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=-1)
            neg_edge_index = negative_sampling(self.edge_index, num_nodes=self.num_nodes)
            neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

            loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() \
                   - torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:03d}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            self.embeddings = model(x, self.edge_index).cpu().numpy()

        print(f"GIN embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_graphsage(self, epochs: int = 100):
        print("GraphSAGE: Training (unsupervised link prediction)...")
        from torch_geometric.nn import SAGEConv

        class GraphSAGE(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        x = torch.eye(self.num_nodes).to(self.device)

        model = GraphSAGE(
            in_channels=self.num_nodes,
            hidden_channels=256,
            out_channels=self.embedding_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = model(x, self.edge_index)

            pos_score = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=-1)
            neg_edge_index = negative_sampling(self.edge_index, num_nodes=self.num_nodes)
            neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

            loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() \
                   - torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:03d}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            self.embeddings = model(x, self.edge_index).cpu().numpy()

        print(f"GraphSAGE embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_gat(self, heads: int = 8, epochs: int = 100, dropout: float = 0.6):

        print("GAT: Training...")
        from torch_geometric.nn import GATConv

        class GAT(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
                super().__init__()
                self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
                self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        model = GAT(
            in_channels=self.num_nodes,
            hidden_channels=256,
            out_channels=self.embedding_dim,
            heads=heads,
            dropout=dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        x = torch.eye(self.num_nodes).to(self.device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = model(x, self.edge_index)

            # Link prediction loss (same as your GCN/GraphSAGE)
            pos_score = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=-1)
            neg_edge_index = negative_sampling(self.edge_index, num_nodes=self.num_nodes)
            neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

            loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean() - \
                   torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:03d}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            self.embeddings = model(x, self.edge_index).cpu().numpy()

        print(f"GAT embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_hope(self, beta: float = 0.5):

        print("HOPE: Computing closed-form solution...")
        from scipy.sparse.linalg import svds

        A = self.adj_csr.astype(np.float64)

        deg = np.array(A.sum(axis=1)).flatten()
        D_inv = sparse.diags(1.0 / (deg + 1e-12))
        M_g = sparse.eye(self.num_nodes) - beta * D_inv @ A   # global proximity polynomial
        M_l = (1 - beta) * A                                 # local proximity

        S = M_l @ sparse.linalg.inv(M_g)   

        S = sparse.csr_matrix((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            if i % 1000 == 0:
                print(f"HOPE solving row {i}/{self.num_nodes}")
            rhs = M_l[:, i].toarray().flatten()
            sol = sparse.linalg.cg(M_g, rhs, tol=1e-6)[0]
            S[i] = sol

        S = S.tocsr()

        U, s, Vt = svds(S, k=self.embedding_dim)
        self.embeddings = np.hstack([U, Vt.T]) @ np.diag(np.sqrt(s))   
        self.embeddings = U @ np.diag(np.sqrt(s))

        print(f"HOPE embeddings: {self.embeddings.shape}")
        return self.embeddings

    def fit_grarep(self, max_order: int = 4):

        print(f"GraRep: Computing up to order {max_order}...")
        from scipy.sparse.linalg import svds

        A = self.adj_csr.astype(np.float64)
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()

        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg != 0)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
        D_inv_sqrt = sparse.diags(deg_inv_sqrt)

        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        transition_embeddings = []

        Ak = sparse.eye(self.num_nodes)  # A^0
        for k in range(1, max_order + 1):
            print(f"  Processing order {k}...")
            Ak = A_norm @ Ak

            log_prob = np.log(Ak.toarray() + 1e-12)
            log_prob[log_prob < 0] = 0   

            U, s, _ = svds(log_prob, k=self.embedding_dim)
            Uk = U @ np.diag(np.sqrt(s))
            transition_embeddings.append(Uk)

        self.embeddings = np.hstack(transition_embeddings)
        print(f"GraRep raw concatenated embeddings: {self.embeddings.shape}")

        if self.embeddings.shape[1] > self.embedding_dim:
            U, s, _ = svds(self.embeddings, k=self.embedding_dim)
            self.embeddings = U @ np.diag(np.sqrt(s))

        print(f"GraRep final embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    # ========================== EVALUATION ==========================
    
    def node_classification(self, test_size: float = 0.2, n_splits: int = 5, dataset: str = 'cora'):
        import pickle
        if self.embeddings is None:
            raise ValueError("Embeddings not trained.")
        
        micro_f1s, macro_f1s = [], []

        for run in range(n_splits):
            with open(f"./splits/{dataset}_node/node_{run}.pkl", 'rb') as f:
                split = pickle.load(f)
            X_train, X_test, y_train, y_test = split['train_idx'], split['test_idx'], split['y_train'], split['y_test']
            X_train = self.embeddings[X_train]
            X_test = self.embeddings[X_test]
            clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            micro_f1s.append(f1_score(y_test, y_pred, average='micro'))
            macro_f1s.append(f1_score(y_test, y_pred, average='macro'))

        micro_mean = np.mean(micro_f1s)
        micro_std  = np.std(micro_f1s)
        macro_mean = np.mean(macro_f1s)
        macro_std  = np.std(macro_f1s)

        print(f"Node Classification (5 runs) - "
            f"Micro-F1: {micro_mean:.4f} ± {micro_std:.4f}, "
            f"Macro-F1: {macro_mean:.4f} ± {macro_std:.4f}")

    def link_prediction(self, test_ratio: float = 0.1, method: str = "node2vec", n_splits: int = 5, dataset: str = 'cora'):

        print(f"\n=== Link Prediction ({method.upper()}) - {n_splits} Full Retrain Runs ===")
        
        aucs, aps = [], []

        for run in range(n_splits):
            import pickle
            with open(f"./splits/{dataset}_link/link_{run}.pkl", 'rb') as f:
                split = pickle.load(f)
            adj_train, pos_train, neg_train, pos_test, neg_test = (
                split['adj_train'], split['pos_train'], split['neg_train'],
                split['pos_test'], split['neg_test']
            )
            temp_fw = GraphEmbeddingFramework(
                adj_csr=adj_train.tocsr(),
                embedding_dim=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                window_size=self.window_size,
                device=self.device
            )
            method_map = {
                'node2vec': lambda: temp_fw.fit_node2vec(p=0.5, q=2, epochs=10),
                'deepwalk': temp_fw.fit_deepwalk,
                'netmf': temp_fw.fit_netmf,
                'sdne': lambda: temp_fw.fit_sdne(epochs=10),
                'gcn': lambda: temp_fw.fit_gcn(epochs=10),
                'gin': lambda: temp_fw.fit_gin(epochs=10),
                'graphsage': lambda: temp_fw.fit_graphsage(epochs=10),
                "gat": lambda: temp_fw.fit_gat(epochs=10),
                'hope': lambda: temp_fw.fit_hope(beta=0.01),
                'grarep': lambda: temp_fw.fit_grarep(max_order=4)
            }
            
            if method not in method_map:
                raise ValueError(f"Method {method} not supported")
            method_map[method]()
            
            embeddings = temp_fw.embeddings
            
            X_train = np.vstack([
                embeddings[pos_train[:, 0]] * embeddings[pos_train[:, 1]],
                embeddings[neg_train[:, 0]] * embeddings[neg_train[:, 1]]
            ])
            y_train = np.hstack([np.ones(len(pos_train)), np.zeros(len(neg_train))])

            X_test = np.vstack([
                embeddings[pos_test[:, 0]] * embeddings[pos_test[:, 1]],
                embeddings[neg_test[:, 0]] * embeddings[neg_test[:, 1]]
            ])
            y_test = np.hstack([np.ones(len(pos_test)), np.zeros(len(neg_test))])

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_score)
            ap = average_precision_score(y_test, y_score)
            aucs.append(auc)
            aps.append(ap)
            print(f"Run {run+1} - AUC: {auc:.4f}, AP: {ap:.4f}")

        auc_mean, auc_std = np.mean(aucs), np.std(aucs)
        ap_mean, ap_std = np.mean(aps), np.std(aps)

        print(f"\nLink Prediction ({method}) - "
            f"AUC: {auc_mean:.4f} ± {auc_std:.4f}, "
            f"AP: {ap_mean:.4f} ± {ap_std:.4f}")

    def visualize_tsne(self, labels: np.ndarray = None, perplexity: int = 30,
                    dataset: str = "cora", method: str = "node2vec", dpi: int = 200):

        if self.embeddings is None:
            raise ValueError("Embeddings not trained.")

        tsne = TSNE(n_components=2,
                    perplexity=min(perplexity, len(self.embeddings)-1),
                    random_state=42,
                    init='pca',          
                    learning_rate='auto')
        emb_2d = tsne.fit_transform(self.embeddings)

        plt.figure(figsize=(7, 7))                   
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                            c=labels,
                            cmap='tab10',
                            s=12,                   
                            alpha=0.8,
                            edgecolors='none')

        plt.axis('off')                               
        plt.xticks([]), plt.yticks([])                 
        plt.gca().set_position([0, 0, 1, 1])           

        save_path = f'tsne_{method}_{dataset}.png'
        plt.savefig(save_path,
                    dpi=dpi,                    
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=False)          
        plt.close()