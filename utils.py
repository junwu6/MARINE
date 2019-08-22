###############################################
# This section of code adapted from tkipf/gcn #
###############################################
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from sklearn import preprocessing


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    G = nx.from_dict_of_lists(graph)
    edges = []
    for (id_i, id_j) in G.edges():
        edges.append([id_i, id_j])
    edges = np.array(edges)
    adj = nx.adjacency_matrix(G)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, edges, features.todense(), labels


def load_wiki_data(dataset_str):
    G = nx.read_edgelist("{}/graph.txt".format(dataset_str), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=np.arange(max(G.nodes()) + 1))
    row, col, values = [], [], []
    with open("{}/tfidf.txt".format(dataset_str), 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            fea = [i for i in lines.split()]
            row.append(int(fea[0]))
            col.append(int(fea[1]))
            values.append(np.float32(fea[2]))
    features = sp.csr_matrix((values, (row, col)), shape=(max(row) + 1, max(col) + 1), dtype=np.float32)
    Label = []
    with open("{}/group.txt".format(dataset_str), 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            Label.append(int(lines.split()[1]))
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(Label)
    edges = np.vstack((adj.tocoo().row, adj.tocoo().col)).transpose()
    return adj, edges, features.toarray(), labels

