# With the below two lines, we may easily make codes run smoothly in tensorflow 2
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()

import tensorflow as tf
import os

from ops import *

# Classes let you abstract away details while programming
class RNN(object): 
    def __init__(self, params, initializer): # Define the input variables

        # Session settings for tf1
        config = tf.ConfigProto(device_count={'GPU':1}) # device_count uses dictionary to allocate the device number of GPU
        config.gpu_options.allow_growth = True # Attempt to allocate only as much GPU memory based on runtime allocations
        config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Take advantage of 50% GPU memory
        self.session = tf.Session(config=config) # Define the session for self/RNN
        
        # # Session settings for tf2
        # config = tf.compat.v1.ConfigProto(device_count = {'GPU':1}) # device_count uses dictionary to allocate the device number of GPU
        # config.gpu_options.allow_growth = True # Attempt to allocate only as much GPU memory based on runtime allocations
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Take advantage of 50% GPU memory
        # self.session = tf.compat.v1.Session(config=config) # Define the session for self/RNN

        self.params = params
        self.model_name = params['model_name']

        # Hyperparameters
        self.learning_rate = params['learning_rate']
        self.decay_rate = params['decay_rate']
        self.decay_step = params['decay_step']
        self.min_grad = params['min_grad']
        self.max_grad = params['max_grad']

        # RNN parameters
        self.max_time_step = params['max_time_step']
        self.cell_layer_num = params['lstm_layer']
        self.dim_embed_unigram = params['dim_embed_unigram']
        self.dim_embed_bigram = params['dim_embed_bigram']
        self.dim_embed_trigram = params['dim_embed_trigram']

        self.dim_embed_fourgram = params['dim_embed_fourgram']

        self.dim_hidden = params['dim_hidden']
        self.dim_rnn_cell = params['dim_rnn_cell']
        self.dim_unigram = params['dim_unigram'] 
        self.dim_bigram = params['dim_bigram'] 
        self.dim_trigram = params['dim_trigram'] 

        self.dim_fourgram = params['dim_fourgram'] 

        self.dim_output = params['dim_output'] # Number of countries
        self.ngram = params['ngram']
        self.ensemble = params['ensemble']
        self.embed = params['embed']
        self.embed_trainable = params['embed_trainable']
        self.checkpoint_dir = params['checkpoint_dir']
        self.initializer = initializer

        # Input data placeholders for tf1
        self.unigram = tf.placeholder(tf.int32, [None, self.max_time_step]) # Maximum time step of RNN; None for #examples
        self.bigram = tf.placeholder(tf.int32, [None, self.max_time_step])
        self.trigram = tf.placeholder(tf.int32, [None, self.max_time_step])

        self.fourgram = tf.placeholder(tf.int32, [None, self.max_time_step])

        self.lengths = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None])
        self.lstm_dropout = tf.placeholder(tf.float32)
        self.hidden_dropout = tf.placeholder(tf.float32)

        # # Input data placeholders for tf2
        # self.unigram = tf.compat.v1.placeholder(tf.int32, shape = tf.TensorShape([None, self.max_time_step]))
        # self.bigram = tf.compat.v1.placeholder(tf.int32, shape = tf.TensorShape([None, self.max_time_step]))
        # self.trigram = tf.compat.v1.placeholder(tf.int32, shape = tf.TensorShape([None, self.max_time_step]))

        # self.fourgram = tf.compat.v1.placeholder(tf.int32, shape = tf.TensorShape([None, self.max_time_step]))

        # self.lengths = tf.compat.v1.placeholder(tf.int32, shape = tf.TensorShape(None))
        # self.labels = tf.compat.v1.placeholder(tf.int32, shape = tf.TensorShape(None))
        # self.lstm_dropout = tf.compat.v1.placeholder(tf.float32)
        # self.hidden_dropout = tf.compat.v1.placeholder(tf.float32)

        # Model settings for tf1
        self.global_step = tf.Variable(0, name="step", trainable=False)
        # When building a machine learning model it is often convenient to distinguish between 
        # variables holding trainable model parameters and other variables such as a step variable used to count training steps.
        self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.global_step,
                self.decay_step, self.decay_rate, staircase=True) # **
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        # If staircase = True, global_step / decay_steps is an integer division and the decayed learning rate follows a staircase function.
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = None
        self.saver = None
        self.losses = None
        self.logits = None

        # # Model settings for tf2
        # self.global_step = tf.Variable(0, name = "step", trainable = False)
        # self.learning_rate = tf.compat.v1.train.exponential_decay(
        #         self.learning_rate, self.global_step,
        #         self.decay_step, self.decay_rate, staircase = True)
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        # self.optimize = None
        # self.saver = None
        # self.losses = None
        # self.logits = None

        # model build for tf1
        self.merged_summary = None
        self.embed_writer = tf.summary.FileWriter(self.checkpoint_dir) # Writes Summary protocol buffers to event files
        self.embed_config = projector.ProjectorConfig()
        # Using the TensorBoard Embedding Projector, you can graphically represent high dimensional embeddings. 
        # This can be helpful in visualizing, examining, and understanding your embedding layers.
        self.projector = None
        self.build_model()
        self.session.run(tf.global_variables_initializer())
       
        # # model build for tf2
        # self.merged_summary = None
        # self.embed_writer = tf.compat.v1.summary.FileWriter(self.checkpoint_dir)
        # # self.embed_writer = tf.summary.create_file_writer(self.checkpoint_dir)        
        # self.embed_config = projector.ProjectorConfig()
        # self.projector = None
        # self.build_model()
        # self.session.run(tf.global_variables_initializer())

        # debug initializer
        '''
        with tf.variable_scope('Unigram', reuse=True):
            unigram_embed = tf.get_variable("embed", [self.dim_unigram, self.dim_embed_unigram], dtype=tf.float32)
            print(unigram_embed.eval(session=self.session))
        '''

    def ngram_logits(self, inputs, length, dim_input, dim_embed=None, 
            initializer=None, trainable=True, scope='ngram'): # Decide which ngram we are going to use
        with tf.variable_scope(scope) as scope: 
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            
            if dim_embed is not None: # dim_embed_0-3
                inputs_embed, self.projector = embedding_lookup(inputs, 
                        dim_input, dim_embed, self.checkpoint_dir, self.embed_config, 
                        draw=True, initializer=initializer, trainable=trainable, scope=scope) # get the embeddings for a specific ngram
                inputs_reshape = rnn_reshape(inputs_embed, dim_embed, self.max_time_step)
                self.projector.visualize_embeddings(self.embed_writer, self.embed_config)
            else:
                inputs_reshape = rnn_reshape(tf.one_hot(inputs, dim_input), dim_input, self.max_time_step) 
                # one-hot if no ngram; turn inputs (indices) into a one-hot matrix (#inputs, dim_input)
            
            outputs = rnn_model(inputs_reshape, length, fw_cell, self.params) # gathered_outputs
            return outputs

    def build_model(self):
        print("## Building an RNN model")

        unigram_logits = self.ngram_logits(inputs=self.unigram, 
                length=self.lengths, 
                dim_input=self.dim_unigram,
                dim_embed=self.dim_embed_unigram if self.embed else None,
                initializer=self.initializer[0],
                trainable=self.embed_trainable,
                scope='Unigram')

        bigram_logits = self.ngram_logits(inputs=self.bigram, 
                length=self.lengths-1, 
                dim_input=self.dim_bigram,
                dim_embed=self.dim_embed_bigram if self.embed else None,
                initializer=self.initializer[1],
                trainable=self.embed_trainable,
                scope='Bigram')
        
        trigram_logits = self.ngram_logits(inputs=self.trigram, 
                length=self.lengths-2, 
                dim_input=self.dim_trigram,
                dim_embed=self.dim_embed_trigram if self.embed else None,
                initializer=self.initializer[2],
                trainable=self.embed_trainable,
                scope='Trigram')

        if self.ensemble:
            total_logits = tf.concat([unigram_logits, bigram_logits, trigram_logits], axis=1)
        elif self.ngram == 1:
            total_logits = unigram_logits
        elif self.ngram == 2:
            total_logits = bigram_logits
        elif self.ngram == 3:
            total_logits = trigram_logits
        else:
            assert True, 'No specific ngram %d'% ngram

        hidden1 = linear(inputs=total_logits, 
                output_dim=self.dim_hidden,
                dropout_rate=self.hidden_dropout,
                activation=tf.nn.relu,
                scope='Hidden1')
        
        logits = linear(inputs=total_logits,
            output_dim=self.dim_output, 
            scope='Output')

        self.logits = logits 
        self.losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=self.labels))

        tf.summary.scalar('Loss', self.losses)
        self.variables = tf.trainable_variables()

        grads = []
        for grad in tf.gradients(self.losses, self.variables):
            if grad is not None:
                grads.append(tf.clip_by_value(grad, self.min_grad, self.max_grad))
            else:
                grads.append(grad)
        self.optimize = self.optimizer.apply_gradients(zip(grads, self.variables), global_step=self.global_step)

        model_vars = [v for v in tf.global_variables()]
        print('model variables', [model_var.name for model_var in tf.trainable_variables()])
        self.saver = tf.train.Saver(model_vars)
        self.merged_summary = tf.summary.merge_all()

    @staticmethod
    def reset_graph():
        tf.reset_default_graph()

    def save(self, checkpoint_dir, step):
        file_name = "%s.model" % self.model_name
        self.saver.save(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model saved", file_name)

    def load(self, checkpoint_dir):
        file_name = "%s.model" % self.model_name
        file_name += "-10800"
        self.saver.restore(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model loaded", file_name)

