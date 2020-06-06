import tensorflow as tf
import os

from tensorflow.contrib.rnn.python.ops.rnn_cell import AttentionCellWrapper # Only used one time in a commented line; https://www.jianshu.com/p/24eaed11bc9b
from tensorflow.contrib.tensorboard.plugins import projector # Not in tf2 anymore...
# # tf2
# from tensorboard.plugins import projector


# tf1
def dropout(x, dropout_rate): # Previous 'keep_prob' is actually dropout_rate
    return tf.nn.dropout(x, keep_prob = 1 - dropout_rate)

# # tf2 https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/dropout
# def dropout(x, dropout_rate):
#     return tf.nn.dropout(x, rate = dropout_rate)


# tf1
def lstm_cell(cell_dim, layer_num, dropout_rate):
    with tf.variable_scope('LSTM_Cell') as scope: # A context manager for defining ops that creates variables (layers)
    # https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-12-scope/
        cell = tf.contrib.rnn.BasicLSTMCell(cell_dim, forget_bias=1.0, activation=tf.tanh, state_is_tuple=True) # #Units of LSTM cell 
        cell = AttentionCellWrapper(cell, attn_length = 10, state_is_tuple=True) # ** Origin is 10; 5 is worse
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1 - dropout_rate)
        return tf.contrib.rnn.MultiRNNCell([cell] * layer_num, state_is_tuple=True)

# # tf2
# def lstm_cell(cell_dim, layer_num, dropout_rate):
#     with tf.compat.v1.variable_scope('LSTM_Cell') as scope:
#         cell = tf.compat.v1.nn.rnn_cell.LSTMCell(cell_dim, forget_bias=1.0, state_is_tuple=True) # None => tanh
#         cell = tfa.seq2seq.AttentionWrapper(cell, attention_layer_size : 10) # ** Not the attention we want in a single layer of LSTM
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1- dropout_rate)
#         return tf.contrib.rnn.MultiRNNCell([cell] * layer_num, state_is_tuple=True)


#tf1
def rnn_reshape(inputs, input_dim, max_time_step): # input_dim => dim_embed
    with tf.variable_scope('Reshape') as scope:
        """
        reshape inputs from [batch_size, max_time_step, input_dim] to [max_time_step * (batch_size, input_dim)]

        :param inputs: inputs of shape [batch_size, max_time_step, input_dim]
        :param input_dim: dimension of input
        :param max_time_step: max of time step

        :return:
            outputs of shape [max_time_step * (batch_size, input_dim)]
        """
        inputs_tr = tf.transpose(inputs, [1, 0, 2]) # (max_time_step, batch_size, input_dim)
        inputs_tr_reshape = tf.reshape(inputs_tr, [-1, input_dim]) # (max_time_step * batch_size, input_dim)
        inputs_tr_reshape_split = tf.split(axis=0, num_or_size_splits=max_time_step,
                value=inputs_tr_reshape) # #Splited rank2 tensors == max_time_step
        return inputs_tr_reshape_split # A list of rank2 tensors

# #tf2
# def rnn_reshape(inputs, input_dim, max_time_step): # input_dim => dim_embed
#     with tf.compat.v1.variable_scope('Reshape') as scope:
#         """
#         reshape inputs from [batch_size, max_time_step, input_dim] to [max_time_step * (batch_size, input_dim)]

#         :param inputs: inputs of shape [batch_size, max_time_step, input_dim]
#         :param input_dim: dimension of input
#         :param max_time_step: max of time step

#         :return:
#             outputs of shape [max_time_step * (batch_size, input_dim)]
#         """
#         inputs_tr = tf.transpose(inputs, [1, 0, 2]) # (max_time_step, batch_size, input_dim)
#         inputs_tr_reshape = tf.reshape(inputs_tr, [-1, input_dim]) # (max_time_step * batch_size, input_dim)
#         inputs_tr_reshape_split = tf.split(axis=0, num_or_size_splits=max_time_step,
#                 value=inputs_tr_reshape) # #Splited rank2 tensors == max_time_step
#         return inputs_tr_reshape_split # A list of rank2 tensors


# tf1
# Hard to change to tf2 since keras is needed
def rnn_model(inputs, input_len, cell, params):
    max_time_step = params['max_time_step']
    dim_rnn_cell = params['dim_rnn_cell']
    with tf.variable_scope('RNN') as scope:
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, sequence_length=input_len, dtype=tf.float32, scope=scope)
        # inputs from rnn_shape; input_len => embedding lengths
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2]) 
        # outputs before is a list of rank2 tensors and an extra dimension is added ((N,a,b)) to be a rank3 tensor; 
        # Then transpose the dimensions as [batch_size, max_time_step, input_dim] in rnn_shape
        spread_len = tf.range(0, tf.shape(input_len)[0]) * max_time_step + (input_len - 1)
        gathered_outputs = tf.gather(tf.reshape(outputs, [-1, dim_rnn_cell]), spread_len)
        return gathered_outputs


#tf1
# Hard to change to tf2 since keras is needed
def bi_rnn_model(inputs, input_len, fw_cell, bw_cell):
    with tf.variable_scope('Bi-RNN') as scope:
        outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs,
                sequence_length=input_len, dtype=tf.float32, scope=scope) # scope defaults to "bidirectional_rnn"
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2]) 
        return outputs


#tf1
def embedding_lookup(inputs, voca_size, embedding_dim, visual_dir, config, draw=False,
        initializer=None, trainable=True, scope='Embedding'):
    with tf.variable_scope(scope) as scope:
        if initializer is not None:
            embedding_table = tf.get_variable("embed",
                    initializer=initializer, trainable=trainable, dtype=tf.float32) # "embed" is the nick name of embedding_table
        else:
            embedding_table = tf.get_variable("embed", [voca_size, embedding_dim],
                    dtype=tf.float32, trainable=trainable)
        inputs_embed = tf.nn.embedding_lookup(embedding_table, inputs) 
        # Looks up ids (inputs) in a list of embedding tensors; instead of doing the matrix multiplication
        print(inputs_embed)

        if draw: # https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin; graphically represent high dimensional embeddings
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_table.name
            embedding.metadata_path = os.path.join(visual_dir, '%s_metadata.tsv'%scope.name)
            return inputs_embed, projector
        else:
            return inputs_embed, None

# #tf2
# def embedding_lookup(inputs, voca_size, embedding_dim, visual_dir, config, draw=False,
#         initializer=None, trainable=True, scope='Embedding'):
#     with tf.compat.v1.variable_scope(scope) as scope:
#         if initializer is not None:
#             embedding_table = tf.compat.v1.get_variable("embed",
#                     initializer=initializer, trainable=trainable, dtype=tf.float32) # "embed" is the nick name of embeddin_table
#         else:
#             embedding_table = tf.compat.v1.get_variable("embed", [voca_size, embedding_dim],
#                     dtype=tf.float32, trainable=trainable)
#         inputs_embed = tf.nn.embedding_lookup(embedding_table, inputs)
#         print(inputs_embed)

#         if draw: # Not sure...
#             embedding = config.embeddings.add()
#             embedding.tensor_name = embedding_table.name
#             embedding.metadata_path = os.path.join(visual_dir, '%s_metadata.tsv'%scope.name)
#             return inputs_embed, projector
#         else:
#             return inputs_embed, None


def mask_by_index(batch_size, input_len, max_time_step):
    with tf.variable_scope('Masking') as scope:
        input_index = tf.range(0, batch_size) * max_time_step + (input_len - 1)
        lengths_transposed = tf.expand_dims(input_index, 1)
        lengths_tiled = tf.tile(lengths_transposed, [1, max_time_step])
        mask_range = tf.range(0, max_time_step)
        range_row = tf.expand_dims(mask_range, 0)
        range_tiled = tf.tile(range_row, [batch_size, 1])
        mask = tf.less_equal(range_tiled, lengths_tiled)
        weight = tf.select(mask, tf.ones([batch_size, max_time_step]),
                           tf.zeros([batch_size, max_time_step]))
        weight = tf.reshape(weight, [-1])
        return weight

# tf1
def linear(inputs, output_dim, dropout_rate=0, regularize_rate=0, activation=None, scope='Linear'): # Origin dropout=1.0 which refers to keep_prob
# inputs => total_logits -> output from LSTM of shape (1,n) where n refers to (the sum of) #LSTM units 
# output_dim => dim_hidden
# dropout_rate => hidden_dropout
         # hidden1 = linear(inputs=total_logits, 
         #        output_dim=self.dim_hidden,
         #        dropout_rate=self.hidden_dropout,
         #        activation=tf.nn.relu,
         #        scope='Hidden1')
    with tf.variable_scope(scope) as scope:
        input_dim = inputs.get_shape().as_list()[-1]
        inputs = tf.reshape(inputs, [-1, input_dim]) # Seems (1,n) again
        weights = tf.get_variable('Weights', [input_dim, output_dim],
                                  initializer=tf.random_normal_initializer())
        variable_summaries(weights, scope.name + '/Weights')
        biases = tf.get_variable('Biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, scope.name + '/Biases')
        if activation is None:
            return dropout((tf.matmul(inputs, weights) + biases), dropout_rate)
        else:
            return dropout(activation(tf.matmul(inputs, weights) + biases), dropout_rate)

# # tf2
# def linear(inputs, output_dim, dropout_rate=0, regularize_rate=0, activation=None, scope='Linear'):
#     with tf.variable_scope(scope) as scope:
#         input_dim = inputs.get_shape().as_list()[-1]
#         inputs = tf.reshape(inputs, [-1, input_dim]) # Seems (1,n) again
#         weights = tf.compat.v1.get_variable('Weights', shape = tf.TensorShape([input_dim, output_dim]),
#                                   initializer = tf.random_normal_initializer())
#         variable_summaries(weights, scope.name + '/Weights')
#         biases = tf.compat.v1.get_variable('Biases', shape = tf.TensorShape([output_dim]),
#                                  initializer = tf.constant_initializer(0.0))
#         variable_summaries(biases, scope.name + '/Biases')
#         if activation is None:
#             return dropout((tf.linalg.matmul(inputs, weights) + biases), dropout_rate)
#         else:
#             return dropout(activation(tf.linalg.matmul(inputs, weights) + biases), dropout_rate) # activation => tf.nn.relu is fine for tf2


# tf1
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean) # name => summaries/mean/Hidden1/Biases
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev) # name => summaries/stddev/Hidden1/Biases
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

# # tf2
# def variable_summaries(var, name):
#     """Attach a lot of summaries to a Tensor."""
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean/' + name, mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev/' + name, stddev)
#         tf.summary.scalar('max/' + name, tf.math.reduce_max(var))
#         tf.summary.scalar('min/' + name, tf.math.reduce_min(var))
#         tf.summary.histogram(name, var)