import scipy.sparse as sp
import numpy as np
import argparse as arg
from model import GraphEmbeddingFramework


arg_parser = arg.ArgumentParser(description="Graph Embedding Framework")
arg_parser.add_argument('--method', type=str, default='node2vec', help='Embedding method to use')
arg_parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use')
args = arg_parser.parse_args()

adj = sp.load_npz(f'data/{args.dataset}/adj.npz')
n_nodes = adj.shape[0]
m_edges = adj.nnz
labels = np.load(f'data/{args.dataset}/labels.npy')
print(f"Dataset: {args.dataset}, Nodes: {n_nodes}, Edges: {m_edges}, Method: {args.method}")


# Framework
framework = GraphEmbeddingFramework(adj_csr=adj, embedding_dim=64)

# Pick any method
if args.method == 'node2vec':
    framework.fit_node2vec(p=0.5, q=2.0)
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == 'deepwalk':
    framework.fit_deepwalk()
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == "netmf":
    framework.fit_netmf()
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == "grarep":
    framework.fit_grarep()
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == 'hope':
    framework.fit_hope(beta=0.01)
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == 'gcn':
    framework.fit_gcn()
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == 'gat':
    framework.fit_gat()
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)
elif args.method == 'gin':
    framework.fit_gin()
    framework.node_classification(dataset=args.dataset)
    framework.link_prediction(method=args.method,dataset=args.dataset)
    framework.visualize_tsne(labels=labels,method=args.method,dataset=args.dataset)

else:
    print(f"Method {args.method} not recognized.")