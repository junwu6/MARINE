from inits import *
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.preprocessing import normalize


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
