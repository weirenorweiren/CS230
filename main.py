import tensorflow as tf
import numpy as np
import pprint

from time import gmtime, strftime
from dataset import get_data, experiment, get_char2vec
from model import RNN


flags = tf.app.flags # Pass the parameters

# *** == we should change to fit our data
# ** == we can change for the model

# Default parameters
flags.DEFINE_integer("train_epoch", 3000, "Epoch to train")
flags.DEFINE_integer("dim_unigram", 82, "Dimension of input, 42 or 82")
flags.DEFINE_integer("dim_bigram", 1876, "Dimension of input, 925 or 1876")
flags.DEFINE_integer("dim_trigram", 14767, "Dimension of input, 8573 or 14767")

flags.DEFINE_integer("dim_output", 127, "Dimension of output, 95 or 127") # *** Number of nationalities

flags.DEFINE_integer("max_time_step", 50, "Maximum time step of RNN") # Origin value is 60, now change to optimal
flags.DEFINE_integer("min_grad", -5, "Minimum gradient to clip")
flags.DEFINE_integer("max_grad", 5, "Maximum gradient to clip")
flags.DEFINE_integer("batch_size", 1000, "Size of batch") # Origin value is 300, now change to optimal

flags.DEFINE_integer("ngram", 3, "Ngram feature when ensemble = False.") # *** 

flags.DEFINE_float("decay_rate", 0.99, "Decay rate of learning rate")
flags.DEFINE_float("decay_step", 100, "Decay step of learning rate")

# Validation hyper parameters
flags.DEFINE_integer("valid_iteration", 250, "Number of validation iteration.")

flags.DEFINE_integer("dim_rnn_cell", 200, "Dimension of RNN cell") # (200, 1) or (200, 1 * dim_embed_gram)
flags.DEFINE_integer("dim_rnn_cell_min", 200, "Minimum dimension of RNN cell")
flags.DEFINE_integer("dim_rnn_cell_max", 399, "Maximum dimension of RNN cell")

flags.DEFINE_integer("dim_hidden", 200, "Dimension of hidden layer") # **(200, 1)
flags.DEFINE_integer("dim_hidden_min", 200, "Minimum dimension of hidden layer")
flags.DEFINE_integer("dim_hidden_max", 399, "Maximum dimension of hidden layer")

flags.DEFINE_integer("dim_embed_unigram", 30, "Dimension of character embedding") # What's the difference between dim_unigram and dim__embed_unigram? 
flags.DEFINE_integer("dim_embed_unigram_min", 10, "Minimum dimension of character embedding") # The former is unigram2indx as reference dictionary
flags.DEFINE_integer("dim_embed_unigram_max", 100, "Maximum dimension of character embedding")

flags.DEFINE_integer("dim_embed_bigram", 100, "Dimension of character embedding")
flags.DEFINE_integer("dim_embed_bigram_min", 30, "Minimum dimension of character embedding")
flags.DEFINE_integer("dim_embed_bigram_max", 200, "Maximum dimension of character embedding")

flags.DEFINE_integer("dim_embed_trigram", 130, "Dimension of character embedding")
flags.DEFINE_integer("dim_embed_trigram_min", 30, "Minimum dimension of character embedding")
flags.DEFINE_integer("dim_embed_trigram_max", 320, "Maximum dimension of character embedding")

flags.DEFINE_integer("lstm_layer", 1, "Layer number of RNN ")
flags.DEFINE_integer("lstm_layer_min", 1, "Mimimum layer number of RNN ")
flags.DEFINE_integer("lstm_layer_max", 1, "Maximum layer number of RNN ")

flags.DEFINE_float("lstm_dropout", 0.5, "Dropout of RNN cell")
flags.DEFINE_float("lstm_dropout_min", 0.3, "Minumum dropout of RNN cell")
flags.DEFINE_float("lstm_dropout_max", 0.8, "Maximum dropout of RNN cell")

flags.DEFINE_float("hidden_dropout", 0.5, "Dropout rate of hidden layer")
flags.DEFINE_float("hidden_dropout_min", 0.3, "Minimum dropout rate of hidden layer")
flags.DEFINE_float("hidden_dropout_max", 0.8, "Maximum dropout rate of hidden layer")

flags.DEFINE_float("learning_rate", 0.0035, "Learning rate of the optimzier") # ** Origin value is 0.01, now change to optimal
flags.DEFINE_float("learning_rate_min", 5e-3, "Minimum learning rate of the optimzier")
flags.DEFINE_float("learning_rate_max", 5e-2, "Maximum learning rate of the optimzier")

# Model settings
flags.DEFINE_boolean("default_params", True, "True to use default params")

flags.DEFINE_boolean("ensemble", True, "True to use ensemble ngram") # *** 

flags.DEFINE_boolean("embed", True, "True to use embedding table")
flags.DEFINE_boolean("embed_trainable", False, "True to use embedding table")
flags.DEFINE_boolean("ethnicity", False, "True to test on ethnicity")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing")
flags.DEFINE_boolean("is_valid", True, "True for validation, False for testing")
flags.DEFINE_boolean("continue_train", False, "True to continue training from saved checkpoint. False for restarting.") # **
flags.DEFINE_boolean("save", False, "True to save") # ***
flags.DEFINE_string("model_name", "default", "Model name, auto saved as YMDHMS")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Directory name to save the checkpoints [checkpoint]")

flags.DEFINE_string("data_dir", "data/raw", "Directory name of input data") # *** We need to change the path to our data here

flags.DEFINE_string("valid_result_path", "result/validation", "Validation result save path")
flags.DEFINE_string("pred_result_path", "result/pred.txt", "Prediction result save path")
flags.DEFINE_string("detail_result_path", "result/detail.txt", "Prediction result save path")

FLAGS = flags.FLAGS


def sample_parameters(params):
    combination = [
            params['dim_hidden'],
            params['dim_rnn_cell'],
            params['learning_rate'],
            params['lstm_dropout'],
            params['lstm_layer'],
            params['hidden_dropout'],
            params['dim_embed_unigram'],
            params['dim_embed_bigram'],
            params['dim_embed_trigram']
    ]

    if not params['default_params']: # If not set too use default params
        combination[0] = params['dim_hidden'] = int(np.random.uniform(
                params['dim_hidden_min'],
                params['dim_hidden_max']) // 50) * 50 
        combination[1] = params['dim_rnn_cell'] = int(np.random.uniform(
                params['dim_rnn_cell_min'],
                params['dim_rnn_cell_max']) // 50) * 50
        combination[2] = params['learning_rate'] = float('{0:.5f}'.format(np.random.uniform( # We could improve with log sampling
                params['learning_rate_min'],
                params['learning_rate_max'])))
        combination[3] = params['lstm_dropout'] = float('{0:.5f}'.format(np.random.uniform( # 5 after the decimal point
                params['lstm_dropout_min'],
                params['lstm_dropout_max'])))
        combination[4] = params['lstm_layer'] = int(np.random.uniform(
                params['lstm_layer_min'],
                params['lstm_layer_max']))
        combination[5] = params['hidden_dropout'] = float('{0:.5f}'.format(np.random.uniform(
                params['hidden_dropout_min'],
                params['hidden_dropout_max'])))
        combination[6] = params['dim_embed_unigram'] = int(np.random.uniform(
                params['dim_embed_unigram_min'],
                params['dim_embed_unigram_max']) // 10) * 10
        combination[7] = params['dim_embed_bigram'] = int(np.random.uniform(
                params['dim_embed_bigram_min'],
                params['dim_embed_bigram_max']) // 10) * 10
        combination[8] = params['dim_embed_trigram'] = int(np.random.uniform(
                params['dim_embed_trigram_min'],
                params['dim_embed_trigram_max']) // 10) * 10

    return params, combination


def main(_):
    # Save default params and set scope
    saved_params = FLAGS.__flags # !!!Not pass the parameters on the Colab when pasting the codes
    if saved_params['ensemble']: # uni + bi + tri
        model_name = 'ensemble'
    elif saved_params['ngram'] == 1:
        model_name = 'unigram'
    elif saved_params['ngram'] == 2:
        model_name = 'bigram'
    elif saved_params['ngram'] == 3:
        model_name = 'trigram'
    else:
        assert True, 'Not supported ngram %d'% saved_params['ngram']
    model_name += '_embedding' if saved_params['embed'] else '_no_embedding' 
    saved_params['model_name'] = '%s' % model_name
    saved_params['checkpoint_dir'] += model_name
    pprint.PrettyPrinter().pprint(saved_params)
    saved_dataset = get_data(saved_params) # Input the passing parameters; Return train_set, valid_set, test_set, dictionary == [idx2unigram, unigram2idx, idx2country, country2ethnicity, idx2bigram, idx2trigram]

    validation_writer = open(saved_params['valid_result_path'], 'a') # Write in a new file if not existing
    validation_writer.write(model_name + "\n")
    validation_writer.write("[dim_hidden, dim_rnn_cell, learning_rate, lstm_dropout, lstm_layer, hidden_dropout, dim_embed]\n")
    validation_writer.write("combination\ttop1\ttop5\tepoch\n")

#     # Run the model
#     for _ in range(saved_params['valid_iteration']):
#         # Sample parameter sets
#         params, combination = sample_parameters(saved_params.copy()) # If not default parameters, then update with initialization
#         dataset = saved_dataset[:]
        
#         # Initialize embeddings
#         uni_init = get_char2vec(dataset[0][0][:], params['dim_embed_unigram'], dataset[3][0]) # Return initializer
#         bi_init = get_char2vec(dataset[0][1][:], params['dim_embed_bigram'], dataset[3][4]) # The first [] is the outermost dimension == train_set or dictionary; [3][i] gives the outermost dimension in dictionary
#         tri_init = get_char2vec(dataset[0][2][:], params['dim_embed_trigram'], dataset[3][5]) # Easy to understand with get_data()
        
#         print(model_name, 'Parameter sets: ', end='')
#         pprint.PrettyPrinter().pprint(combination)
        
#         rnn_model = RNN(params, [uni_init, bi_init, tri_init])
#         top1, top5, ep = experiment(rnn_model, dataset, params)
        
#         validation_writer.write(str(combination) + '\t')
#         validation_writer.write(str(top1) + '\t' + str(top5) + '\tEp:' + str(ep) + '\n')

#     validation_writer.close()

# if __name__ == '__main__':
#     tf.app.run()

