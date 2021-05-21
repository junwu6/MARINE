from __future__ import print_function
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import sklearn.utils.linear_assignment_ as la


def get_roc_score(edges_pos, edges_neg, emb=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    adj_rec = np.dot(emb, emb.T)
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def link_prediction_ROC(edges_pos, embeddings):
    np.random.seed(seed=0)
    print("Performing the link prediction..................")
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(edges_pos):
        idx_i = np.random.randint(0, embeddings.shape[0])
        idx_j = np.random.randint(0, embeddings.shape[0])
        if idx_i == idx_j:
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    roc_score, ap_score = get_roc_score(edges_pos, test_edges_false, emb=embeddings)
    return ap_score


def run_svm(tr_embeds, tr_labels, ts_embeds, ts_labels):
    rnd = np.random.randint(2019)
    model = LinearSVC(random_state=rnd)
    model.fit(tr_embeds, np.argmax(tr_labels, axis=1))
    y_test_pred = model.predict(ts_embeds)
    return accuracy_score(np.argmax(ts_labels, axis=1), y_test_pred), \
           f1_score(np.argmax(ts_labels, axis=1), y_test_pred, average='macro'), \
           f1_score(np.argmax(ts_labels, axis=1), y_test_pred, average='micro')


def node_classification(embeddings, labels, indice, norm=True):
    print("Performing the node classification..................")
    if norm:
        embeddings = normalize(embeddings)
    num_data = embeddings.shape[0]
    acc_all = np.zeros(shape=[1, 9])
    run_times = 10
    for k in range(1, 10):
        print("The training data set is: {}%".format(10*k))
        run_acc = []
        for j in range(run_times):
            idx_train = indice[:k * num_data // 10]
            idx_test = indice[k * num_data // 10:]
            train_labels = np.array([labels[i] for i in idx_train])
            test_labels = np.array([labels[i] for i in idx_test])
            train_embeds = embeddings[np.array(idx_train), :]
            test_embeds = embeddings[np.array(idx_test), :]
            acc, f1_macro, f1_micro = run_svm(train_embeds, train_labels, test_embeds, test_labels)
            run_acc.append(acc)
        acc_all[0, k-1] = np.mean(run_acc)
    return acc_all


def best_map(l1, l2):
    """
    Please check https://github.com/jundongl/scikit-feature/blob/master/skfeature/utility/unsupervised_evaluation.py
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)

    label1 = np.unique(l1)
    n_class1 = len(label1)

    label2 = np.unique(l2)
    n_class2 = len(label2)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = la.linear_assignment(-G)

    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)


def node_clustering(embeddings, labels, n_class, norm=True):
    print("Performing the node clustering..................")
    if norm:
        embeddings = normalize(embeddings)
    labels = np.argmax(labels, axis=1)

    run_times = 10
    NMI, AC, RI = [], [], []
    for _ in range(run_times):
        rnd = np.random.randint(2019)
        kmeans = KMeans(n_clusters=n_class, random_state=rnd).fit(embeddings)
        pred_labels = kmeans.labels_

        # calculate NMI
        nmi_score = normalized_mutual_info_score(labels, pred_labels)
        NMI.append(nmi_score)
        # calculate AC
        y_permuted_predict = best_map(labels, pred_labels)
        acc = accuracy_score(labels, y_permuted_predict)
        AC.append(acc)
        # calculate Rand index
        ri_score = adjusted_rand_score(labels, pred_labels)
        RI.append(ri_score)
    return np.mean(NMI), np.mean(AC), np.mean(RI)
