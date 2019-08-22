from inits import *


def forward_pass(embX, scope, n_emb, n_fea):
    with tf.variable_scope(scope):
        W1 = tf.get_variable("W_hidden_1", [n_emb, n_fea])
        b1 = tf.get_variable("b_hidden_1", [n_fea])
        output_layer = tf.nn.relu(tf.add(tf.matmul(embX, W1), b1))
    return output_layer


def initialize_model(scope_name, n_emb, n_fea, learning_rate, nodeCount, lamb, eta, batchsize):
    node_heads = tf.placeholder(tf.int32, [batchsize])
    node_tails = tf.placeholder(tf.int32, [batchsize])

    heads_featX = tf.placeholder(tf.float32, [None, n_fea])
    tails_featX = tf.placeholder(tf.float32, [None, n_fea])

    adj_score = tf.placeholder(tf.float32, [1, batchsize])
    d_score = tf.placeholder(tf.float32, [None, 1])

    with tf.variable_scope(scope_name) as scope:
        # Create parameters
        initializer = tf.contrib.layers.xavier_initializer()
        tf.get_variable_scope().set_initializer(initializer)

        embeddings = tf.get_variable("emb", [nodeCount, n_emb])
        heads_emb = tf.gather(embeddings, node_heads)
        tails_emb = tf.gather(embeddings, node_tails)

        # forward pass for head nodes
        heads_prox = forward_pass(heads_emb, scope, n_emb, n_fea)

        # Siamese neural network: reuse the NN defined
        tf.get_variable_scope().reuse_variables()

        # forward pass for tail nodes
        tails_prox = forward_pass(tails_emb, scope, n_emb, n_fea)

        heads_loss = tf.reduce_sum(tf.multiply(heads_featX - heads_prox, heads_featX - heads_prox), axis=1)
        tails_loss = tf.reduce_sum(tf.multiply(tails_featX - tails_prox, tails_featX - tails_prox), axis=1)

        # preserving proximity
        prox_loss = tf.reduce_mean(heads_loss) + tf.reduce_mean(tails_loss)

        # preserving ranking
        heads_score = tf.gather(d_score, node_heads)
        tails_score = tf.gather(d_score, node_tails)
        norm_heads_emb = tf.multiply(heads_score, heads_emb)
        norm_tails_emb = tf.multiply(tails_score, tails_emb)
        r_loss = tf.reduce_sum(tf.multiply(norm_heads_emb - norm_tails_emb, norm_heads_emb - norm_tails_emb),
                               axis=1, keepdims=True)
        ranking_loss = tf.squeeze(tf.matmul(adj_score, r_loss))

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])

        # define costs
        cost = prox_loss + lamb * ranking_loss + lossL2 * eta

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    inits = (init, optimizer, cost, node_heads, node_tails, heads_featX, tails_featX, adj_score, d_score,
             embeddings)
    return inits


def train_model(init, optimizer, cost, node_heads, node_tails,
                heads_featX, tails_featX, feats, adj_score, d_score, adj, embeddings, scope_name, epoch,
                graph, batchsize, gpu_fraction=0.20, print_every_epoch=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    inv_d_value = 1.0/np.sum(adj, axis=1)
    with tf.variable_scope(scope_name, reuse=True) as scope:
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.InteractiveSession(config=tf_config)
        with sess.as_default():
            sess.run(init)
            for n in range(epoch):
                obj = 0.0
                indexes = np.random.permutation(len(graph))
                num_mini_batch = int(len(graph) / batchsize)
                for i in range(num_mini_batch):
                    inds = indexes[i * batchsize: (i + 1) * batchsize]
                    edges = graph[inds].astype(np.int32)
                    heads = edges[:, 0]
                    tails = edges[:, 1]
                    batch_score = np.ones(shape=[1, batchsize], dtype=np.float32)

                    feed_dict = {
                        node_heads: heads,
                        node_tails: tails,
                        heads_featX: feats[heads, :],
                        tails_featX: feats[tails, :],
                        adj_score: batch_score,
                        d_score: inv_d_value
                    }

                    _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                    obj += c

                if print_every_epoch and ((n + 1) % print_every_epoch == 0):
                    print('\tepoch: %d; obj: %.7f' % (n + 1, obj / num_mini_batch))

            final_embeddings = sess.run(embeddings)

    return final_embeddings


def run_model(graph, adj, feats, lamb, eta, n_emb, learning_rate, epoch,
              gpu_fraction=0.0, batchsize=1024, print_every_epoch=1,
              scope_name='default'):
    nodeCount, n_fea = feats.shape
    (init, optimizer, cost, node_heads, node_tails, heads_featX, tails_featX, adj_score, d_score, embeddings) = \
        initialize_model(scope_name, n_emb, n_fea, learning_rate, nodeCount, lamb, eta, batchsize)

    embeddings = train_model(init, optimizer, cost, node_heads, node_tails,
                heads_featX, tails_featX, feats, adj_score, d_score, adj, embeddings, scope_name, epoch,
                graph, batchsize, gpu_fraction=gpu_fraction, print_every_epoch=print_every_epoch)

    return embeddings
