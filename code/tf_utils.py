import numpy as np
import tensorflow as tf
import time


def next_batch(X, Y, batch_size=1, shuffle=True):
    """
    Generator that supplies mini batches during training
    """
    
    n_data = len(Y)

    if shuffle:
        idx = np.random.permutation(n_data)
    else:
        idx = range(n_data)

    # Timeseries or vectorial data?
    if len(X.shape) == 3:
        X = X[:, idx, :]
    else:
        X = X[idx, :]

    Y = Y[idx, :]

    n_batches = n_data//batch_size

    for i in range(n_batches):
        if len(X.shape) == 3:
            X_batch = X[:, i*batch_size:(i + 1)*batch_size, :]
        else:
            X_batch = X[i*batch_size:(i + 1)*batch_size, :]

        Y_batch = Y[i*batch_size:(i + 1)*batch_size]

        yield X_batch, Y_batch


def train_tf_model(model_name, X, Y, Xte, Yte, batch_size,
                   num_epochs, nn_input, keep_prob, logits, nn_output,
                   w_l2, learning_rate, p_drop):
    
    """
    Train a neural network using crossentropy loss with L2 regularization,
    dropout and Adam optimizer.
    Network structure is specified by the "logits" input argument.
    
    Parameters:
        model_name: name of the file to save the trained model
        X: input data. Can be a matrix [N,V] or a tensor [T,N,V]
        Y: lables of the inputs
        Xte: test data
        Yte: labels of the
        batch_size: size of each mini batch
        num_epochs: number of training iterations
        nn_input: TF placeholder for input of the computational graph
        keep_prob: TF placeholder for probability of keeping connections in dropout
        logits: specify the output of the forward pass, hence the network architecture
        nn_output: TF placeholder for target outputs
        w_l2: weight of the L2 regularization
        learning_rate: learning rate in the gradient descent optimization
        p_drop: probability of keeping connections in dropout
    """
    
    with tf.Session() as sess:
        min_val_loss = np.infty
        model_name = '../models/' + model_name+str(time.strftime("_%H%M%S-%d%m"))
        saver = tf.train.Saver()

        # Classification loss
        class_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits,
                        labels=nn_output
                        )
                )

        # L2 loss
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        # Optimizer
        tot_loss = tf.add(class_loss, w_l2*reg_loss, name='tot_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(tot_loss)

        # Compute predicted class and initialize parameters
        pred_class = tf.argmax(logits, axis=1, name='pred_class')
        sess.run(tf.global_variables_initializer())
        

        # ================= TRAINING =================
        try:
            for t in range(num_epochs):

                for X_batch, Y_batch in next_batch(X=X,
                                                   Y=Y,
                                                   batch_size=batch_size,
                                                   shuffle=True):
                    fdtr = {nn_input: X_batch,
                            nn_output: Y_batch,
                            keep_prob: p_drop}

                    _, train_class_loss = sess.run([train_op, class_loss], fdtr)


                # validate training progress on the whole dataset
                if t % 100 == 0:
                    fdvs = {nn_input: X,
                            nn_output: Y,
                            keep_prob: 1.0}
                                            
                    val_loss = sess.run(tot_loss, fdvs)

                    # Save model yielding best results
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss                            
                        saver.save(sess, model_name, write_meta_graph=False)

        except KeyboardInterrupt:
            print('training interrupted')
            
        # ================= TEST =================
        saver.restore(sess, model_name)
        fdte = {nn_input: Xte,
                nn_output: Yte,
                keep_prob: 1.0}
        te_pred_class = sess.run(pred_class, fdte)


    return te_pred_class
