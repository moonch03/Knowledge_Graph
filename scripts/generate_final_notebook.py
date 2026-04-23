import json
import os

def generate_final_notebook():
    output_path = r"C:\Users\USER\Graph\notebooks\Graph_Network.ipynb"
    
    cells = []
    
    # 1. Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import networkx as nx\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from node2vec import Node2Vec\n",
            "from sklearn.manifold import TSNE\n",
            "import torch\n",
            "import torch.nn.functional as F\n",
            "from torch_geometric.nn import SAGEConv\n",
            "from torch_geometric.data import Data\n",
            "from matplotlib.lines import Line2D\n",
            "\n",
            "%matplotlib inline"
        ]
    })
    
    # 2. Noordin Data Loading
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Noordin Top Terrorist Network Analysis"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load Noordin Top JSONL data\n",
            "noordin_path = '../data/noordin.jsonl'\n",
            "G_noordin = nx.Graph()\n",
            "\n",
            "if os.path.exists(noordin_path):\n",
            "    with open(noordin_path, 'r', encoding='utf-8') as f:\n",
            "        for line in f:\n",
            "            obj = json.loads(line)\n",
            "            if obj['type'] == 'node':\n",
            "                G_noordin.add_node(obj['id'], **obj['properties'])\n",
            "            elif obj['type'] == 'edge':\n",
            "                G_noordin.add_edge(obj['source'], obj['target'], **obj['properties'])\n",
            "    print(f\"Noordin Graph: {G_noordin.number_of_nodes()} nodes, {G_noordin.number_of_edges()} edges.\")\n",
            "else:\n",
            "    print(\"Error: noordin.jsonl not found in data directory.\")"
        ]
    })
    
    # 3. Noordin Metrics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def calculate_metrics(G):\n",
            "    degree = nx.degree_centrality(G)\n",
            "    betweenness = nx.betweenness_centrality(G)\n",
            "    closeness = nx.closeness_centrality(G)\n",
            "    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)\n",
            "    clustering = nx.clustering(G)\n",
            "    \n",
            "    data = []\n",
            "    for node in G.nodes():\n",
            "        data.append({\n",
            "            \"ID\": node,\n",
            "            \"Name\": G.nodes[node].get('name', node),\n",
            "            \"Degree\": degree[node],\n",
            "            \"Betweenness\": betweenness[node],\n",
            "            \"Closeness\": closeness[node],\n",
            "            \"Eigenvector\": eigenvector[node],\n",
            "            \"Clustering\": clustering[node]\n",
            "        })\n",
            "    return pd.DataFrame(data).sort_values(\"Degree\", ascending=False)\n",
            "\n",
            "noordin_metrics = calculate_metrics(G_noordin)\n",
            "noordin_metrics.head(10)"
        ]
    })
    
    # 4. Montreal Data Loading
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Montreal Gangs Network Analysis"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load Montreal Gang JSONL data\n",
            "montreal_path = '../data/montreal.jsonl'\n",
            "G_montreal = nx.Graph()\n",
            "\n",
            "if os.path.exists(montreal_path):\n",
            "    with open(montreal_path, 'r', encoding='utf-8') as f:\n",
            "        for line in f:\n",
            "            obj = json.loads(line)\n",
            "            if obj['type'] == 'node':\n",
            "                G_montreal.add_node(obj['id'], **obj['properties'])\n",
            "            elif obj['type'] == 'edge':\n",
            "                G_montreal.add_edge(obj['source'], obj['target'], **obj['properties'])\n",
            "    print(f\"Montreal Graph: {G_montreal.number_of_nodes()} nodes, {G_montreal.number_of_edges()} edges.\")\n",
            "else:\n",
            "    print(\"Error: montreal.jsonl not found in data directory.\")"
        ]
    })
    
    # 5. Montreal Metrics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "montreal_metrics = calculate_metrics(G_montreal)\n",
            "montreal_metrics.head(10)"
        ]
    })
    
    # 6. Embeddings Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Advanced Graph Embeddings (Node2Vec & GraphSAGE)"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def run_embeddings_pipeline(G, name):\n",
            "    print(f\"\\n--- Processing {name} ---\")\n",
            "    # 1. Node2Vec\n",
            "    n2v = Node2Vec(G, dimensions=64, walk_length=20, num_walks=100, p=1, q=1, workers=4, quiet=True)\n",
            "    model = n2v.fit(window=10, min_count=1, batch_words=4)\n",
            "    \n",
            "    # 2. t-SNE Visualization\n",
            "    nodes = list(G.nodes())\n",
            "    X = np.array([model.wv[n] for n in nodes])\n",
            "    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(nodes)-1))\n",
            "    X_2d = tsne.fit_transform(X)\n",
            "    \n",
            "    plt.figure(figsize=(10, 8))\n",
            "    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, s=100)\n",
            "    for i, node in enumerate(nodes):\n",
            "        if i < 15: # Label top 15 nodes\n",
            "            plt.text(X_2d[i, 0]+0.1, X_2d[i, 1]+0.1, node, fontsize=9)\n",
            "    plt.title(f\"Node2Vec Embeddings: {name}\")\n",
            "    plt.show()\n",
            "    \n",
            "    return X\n",
            "\n",
            "noordin_embeddings = run_embeddings_pipeline(G_noordin, \"Noordin Top\")"
        ]
    })
    
    # 7. GraphSAGE
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class SAGEModel(torch.nn.Module):\n",
            "    def __init__(self, in_dim, hidden_dim, out_dim):\n",
            "        super().__init__()\n",
            "        self.conv1 = SAGEConv(in_dim, hidden_dim)\n",
            "        self.conv2 = SAGEConv(hidden_dim, out_dim)\n",
            "    def forward(self, x, edge_index):\n",
            "        x = self.conv1(x, edge_index).relu()\n",
            "        x = self.conv2(x, edge_index)\n",
            "        return x\n",
            "\n",
            "def run_sage(G, x_features):\n",
            "    nodes = list(G.nodes())\n",
            "    node_to_idx = {n: i for i, n in enumerate(nodes)}\n",
            "    edge_index = []\n",
            "    for u, v in G.edges():\n",
            "        edge_index.append([node_to_idx[u], node_to_idx[v]])\n",
            "        edge_index.append([node_to_idx[v], node_to_idx[u]])\n",
            "    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()\n",
            "    x = torch.tensor(x_features, dtype=torch.float)\n",
            "    \n",
            "    model = SAGEModel(in_dim=x.shape[1], hidden_dim=32, out_dim=16)\n",
            "    model.eval()\n",
            "    with torch.no_grad():\n",
            "        embeddings = model(x, edge_index)\n",
            "    return embeddings\n",
            "\n",
            "noordin_sage = run_sage(G_noordin, noordin_embeddings)\n",
            "print(f\"GraphSAGE Embeddings Shape: {noordin_sage.shape}\")"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Final notebook generated at {output_path}")

if __name__ == "__main__":
    generate_final_notebook()
