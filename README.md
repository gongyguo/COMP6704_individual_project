# COMP6704 Individual Project: Graph Representation Learning  

**Graph Embedding as Optimization: From Random Walks to Neural Message Passing**

---

### Overview
This repository contains the complete code and experimental results for the COMP6704 individual assignment on Graph Embedding from an Optimization Perspective.

We implement and compare **seven representative methods** across three major paradigms:
1. Random Walk-based: **DeepWalk**, **Node2Vec**
2. Matrix Factorization-based: **GraRep**,**NetMF**
3. Graph Neural Networks: **GCN**, **GAT**, **GIN**

All methods are evaluated on four benchmark datasets (**Wiki, Cora, Citeseer, BlogCatalog**) for **node classification** and **link prediction**, with fixed embedding dimension = 128.

---

### Project Structure
```
COMP6704_individual_project/
├── main.py                  # Run experiments for different methods & visualization
├── model.py                 # Core implementation and train/test splits generation
├── preprocess.py            # Dataset loading and preprocessing
├── bash.sh                  # Running script
├── environment.yml          # Experimental environment
├── data/                    # Graph datasets
├── splits/                  # Pre-generated train/test splits (5 runs)
│   ├── cora_node/           # Node classification splits on cora
│   └── cora_link/           # Link prediction splits on cora
├── plot/                    # t-SNE plots and result figures
└── README.md
```
