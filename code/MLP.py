import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.layers import maxout
from kafnets import kaf


def fc_layer(input_, in_dim, size, init, w_l2):
    """
    Define and evaluate a fully connected layer.
    
    Parameters:
        input_: TF placeholder for fc layer input
        in_dim: dimension of the input
        size: number of neurons in the layer
        init: initialization scheme for the weights {"he", "xav"}
        w_l2: strength of the L2 regularization on the weights
    """
    
    if init == 'he':
        init_type = tf.keras.initializers.he_normal(seed=None)
    elif init == 'xav':
        init_type = tf.contrib.layers.xavier_initializer()
    else:
        raise RuntimeError('Unknown initializer, can be {"he", "xav"}')

    W = tf.get_variable('W', shape=(in_dim, size), 
                        dtype=tf.float32,
                        initializer=init_type, 
                        regularizer=tf.contrib.layers.l2_regularizer(w_l2)
                        )

    b = tf.get_variable('bias', shape=(size,),
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer)

    fc_output = tf.add(tf.matmul(input_, W), b)

    return fc_output


def build_network(input_, 
                  input_dim,
                  output_dim,
                  net_layout, 
                  keep_prob, 
                  nonlinearity='relu', 
                  init='he', 
                  w_l2=0.0001):
    """
    Build a parametric MLP.
    
    Parameters:
        input_: TF placeholder for MLP input
        input_dim: dimension of the input
        output_dim: dimension of the output
        net_layout: list specifying number of layers and neurons in the MLP
        keep_prob: TF placeholder for probability of keeping connections in dropout
        nonlinearity: type of activation function {'relu', 'tanh', 'maxout', 'kaf', 'sigmoid','linear'}
        init: initialization scheme for the weights {"he", "xav"}
        w_l2: strength of the L2 regularization on the weights
    """

    in_dim = input_dim

    for i, neurons in enumerate(net_layout):
        with tf.variable_scope('h{}'.format(i)):
            if nonlinearity == 'relu':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
                layer_out = tf.nn.relu(layer_out)
            elif nonlinearity == 'maxout':
                K = 5
                layer_out = fc_layer(input_, in_dim, neurons*K, init, w_l2)
                layer_out = maxout(layer_out, neurons)
            elif nonlinearity == 'kaf':
                layer = fc_layer(input_, in_dim, neurons)
                layer = kaf(layer, 'kaf_' + str(i))
            elif nonlinearity == 'sigmoid':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
                layer_out = tf.nn.sigmoid(layer_out)
            elif nonlinearity == 'tanh':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
                layer_out = tf.nn.tanh(layer_out)
            elif nonlinearity == 'linear':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
            else:
                raise RuntimeError('Unknown nonlinearity, can be {"relu", "maxout", "sigmoid", "tanh", "linear"}')

            layer_out = tf.nn.dropout(layer_out, keep_prob=keep_prob)

            input_ = layer_out
            in_dim = neurons
            
    layer_out = fc_layer(input_, in_dim, output_dim, init, w_l2)

    return layer_out


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
        X = X[:, idx, :] # assuming time-major
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


# build computational graph
def make_MLPgraph( input_dim,
                   output_dim,
                   mlp_layout=[10,10], # number of neurons per layer
                   nonlinearity='relu',
                   init='he',
                   learning_rate=0.001,
                   w_l2=0.001,
                   max_gradient_norm=1.0,
                   seed=None):
    
    # initialize computational graph
    g = tf.Graph()
    with g.as_default():
        
        tf.set_random_seed(seed) # only affects default graph
    
        # placeholders
        mlp_inputs = tf.placeholder(shape=(None,input_dim), dtype=tf.float32, name='mlp_inputs')
        mlp_output = tf.placeholder(shape=(None,output_dim),dtype=tf.float32,name='mlp_output')
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
                
        logits = build_network(mlp_inputs,
                               input_dim,
                               output_dim,
                               mlp_layout,
                               keep_prob,
                               nonlinearity,
                               init,
                               w_l2)
        
        # Classification loss
        class_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits,
                        labels=mlp_output
                        )
                )

        # L2 loss
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # tot loss
        tot_loss = tf.add(class_loss, reg_loss, name='tot_loss')

        # Calculate and clip gradients
        parameters = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = tf.gradients(tot_loss, parameters)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))
        
        
        # Print trainable parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            vshape = variable.get_shape()
            variable_parametes = 1
            for dim in vshape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print('Total parameters: {}'.format(total_parameters))

        # put in a collection placeholders and operations to call later 
        g.add_to_collection('MLP_collection', mlp_inputs) #0
        g.add_to_collection('MLP_collection', keep_prob) #1
        g.add_to_collection('MLP_collection', mlp_output) #2
        g.add_to_collection('MLP_collection', update_step) #3
        g.add_to_collection('MLP_collection', class_loss) #4
        g.add_to_collection('MLP_collection', reg_loss) #5
        g.add_to_collection('MLP_collection', tot_loss) #6
        g.add_to_collection('MLP_collection', logits) #7
                        
    print('graph building is done')
    
    return g


def trainMLP(X, 
            Y, 
            Xval,
            Yval,
            batch_size=25, 
            num_epochs=1000, 
            dropout_prob=0.0,
            save_id='default',
            input_graph=None):
    
    if input_graph is None:
        raise RuntimeError('an input_graph must be provided')
        
#        input_graph.as_default()
            
    with tf.Session(graph=input_graph) as sess:
                                 
        # restore placeholders and operations                            
        mlp_inputs = input_graph.get_collection('MLP_collection')[0]
        keep_prob = input_graph.get_collection('MLP_collection')[1]
        mlp_output = input_graph.get_collection('MLP_collection')[2]
        update_step = input_graph.get_collection('MLP_collection')[3]
        class_loss = input_graph.get_collection('MLP_collection')[4]
        reg_loss = input_graph.get_collection('MLP_collection')[5]
        tot_loss = input_graph.get_collection('MLP_collection')[6]
                    
        # initialize trainable variables 
        print('init vars')
        sess.run(tf.global_variables_initializer())

        # initialize training stuff
        time_tr_start = time.time()
        loss_track = []
        min_val_loss = np.infty
        saver = tf.train.Saver() 
        model_name = '../models/MLP_model_'+save_id
        
        print('training start')            
        try:
            for t in range(num_epochs):
                for X_batch, Y_batch in next_batch(X, Y, batch_size, True):
                    fdtr = {mlp_inputs: X_batch,
                            mlp_output: Y_batch,
                            keep_prob: 1.0 - dropout_prob}

                    _, train_loss = sess.run([update_step, tot_loss], fdtr)                        
                    loss_track.append(train_loss)

                # check training progress on the validation set
                if t % 100 == 0:
                    fdvs = {mlp_inputs: Xval,
                            mlp_output: Yval,
                            keep_prob: 1.0}
                    
                    (val_tot_loss,
                     val_class_loss,
                     val_reg_loss) = sess.run([tot_loss,
                                               class_loss,
                                               reg_loss], fdvs)
    
                    print("totL: %.3f, classL: %.3f, reg_loss: %.3f"%(val_tot_loss, val_class_loss, val_reg_loss))

                    # Save model yielding the lowest loss on validation
                    if val_tot_loss < min_val_loss:
                        min_val_loss = val_tot_loss
                        saver.save(sess, model_name)

        except KeyboardInterrupt:
            print('training interrupted')

    ttime = (time.time()-time_tr_start)/60
    print("tot time:{}".format(ttime))
    
    return loss_track


def testMLP(Xte, Yte, save_id='default'):
            
    tf.reset_default_graph()
    with tf.Session() as sess:
        
        model_name = '../models/MLP_model_'+save_id
        saver = tf.train.import_meta_graph(model_name+'.meta', clear_devices=True) 
                
        # restore op                 
        mlp_inputs = tf.get_collection('MLP_collection')[0]
        keep_prob = tf.get_collection('MLP_collection')[1]
        mlp_output = tf.get_collection('MLP_collection')[2]
        class_loss = tf.get_collection('MLP_collection')[4]
        logits = tf.get_collection('MLP_collection')[7]
            
        # restore weights
        saver.restore(sess, model_name)
                    
        # evaluate model on the test set
        fdte = {mlp_inputs: Xte, 
                mlp_output: Yte,
                keep_prob: 1.0}
        te_class_loss, te_logits = sess.run([class_loss, logits], fdte)      
        
        te_pred = np.argmax(te_logits, axis=1)
        
        print('class_loss = %.3f'%te_class_loss)
    
    return te_class_loss, te_pred