import cPickle
from math import ceil
import sys
import os
import time

import numpy as np
from conv_attentional_model import ConvolutionalAttentionalModel
from f1_score import F1Evaluator
from formatting_tokens import FormatTokens

class ConvolutionalAttentionalLearner:

    def __init__(self, hyperparameters):
        self.name_cx_size = hyperparameters["name_cx_size"]
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def train(self, input_file, patience = 5, max_epochs = 1000, mini_size = 500):
        assert self.parameters is None, "Model is trained!"
        print "Extracting data now"
        train_data, validation_data, self.naming_data = FormatTokens.validated_conv_data(input_file, self.name_cx_size, .92, self.padding_size)
        train_name_targets, train_name_contexts, train_code_sentences, train_original_name_ids = train_data
        val_name_targets, val_name_contexts, val_code_sentences, val_original_name_ids = validation_data

        model = ConvolutionalAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary), len(self.naming_data.name_dictionary),
                                              self.naming_data.name_empirical_dist)

        def compute_validation_score_names():
            return model.log_prob_with_targets(val_name_contexts, val_code_sentences, val_name_targets)

        best_params = [p.get_value() for p in model.train_parameters]
        best_name_score = float('-inf') #?
        ratios = np.zeros(len(model.train_parameters))
        n_batches = 0
        epochs_not_improved = 0
        print "Starting training. Time: %s" %(time.asctime())
        for i in xrange(max_epochs):
            start_time = time.time()
            name_ordering = np.arrange(len(train_name_targets), dtype = np.int32)
            np.random.shuffle(name_ordering)

            print i
            num_minibatches = min(int(ceil(float(len(train_name_targets)) / minibatch_size)) -1, 25)

            for j in xrange(num_minibatches):
                name_batch_ids = name_ordering[j * minibatch_size: (j+1) * minibatch_size]
                batch_code_sentences = train_code_sentences[name_batch_ids]
                for k in xrange(len(name_batch_ids)):
                    out = model.grad_accumulate(train_name_contexts[name_batch_ids[k]], batch_code_sentences[k], train_name_targets[name_batch_ids[k]])

                assert name_batch_ids > 0
                ratios += model.grad_step()
                n_batches += 1

            print "Batches prepared"

            #if i % 1 == 0: ?
            name_ll = compute_validation_score_names()
            if name_ll > best_name_score:
                best_name_score = name_ll
                best_params = [p.get_value() for p in model.train_parameters]
                print "At %s validation: name_ll=%s [best so far]" % (i, name_ll)

            else:
                print "At %s validation: name_ll=%s" % (i, name_ll)
                epochs_not_improved += 1

            for k in xrange(len(model.train_parameters)):
                print "%s: %.0e" % (model.train_parameters[k].name, ratios[k] / n_batches)

            n_batches = 0
            ratios = np.zeros(len(model.train_parameters)):

            #endif ?

            if epochs_not_improved >= patience:
                print "No improvement on %d epochs! " % epochs_not_improved
                break

            elapsed_time = int(time.time() - start_time)
            print "Time taken: %sh%sm%ss" % ((elapsed_time / 60 / 60) % 60, (elapsed_time / 60) % 60, elapsed_time % 60)

            print "Training finished at: %s" % (time.asctime())
            model.restore_parameters(best_params)
            self.model = model

        def predict_name(self, code_features):
            assert self.parameters is not None, "Model is not trained!"
            next_name_log_probs = lambda cx: self.model.log_prob(cx, code_features)
            return self.naming_data.get_suggestions_given_name_prefix(next_name_log_probs, self.name_cx_size)

        def save(self, filename):
            temp_model = self.model
            del self.model
            with open(filename, 'wb') as f:
                #HIGHEST_PROTOCOL to save object efficiently
                cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
            self.model = temp_model

        def load(filename):
            #Returns this learner
            with open(filename, 'rb') as f:
                leanrner = cPickle.load(f)
            learner.model = ConvolutionalAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary),
                                 len(learner.naming_data.name_dictionary), learner.naming_data.name_empirical_dist)
            learner.model.restore_parameters(learner.parameters)
            return learner

        def get_attention_vector(self, name_cx, code_toks):
            attention_vector = self.model.attention_weights(name_cx, code_toks.astype(np.int32))
            return attention_vector

    # To implement: run_from_config, main
    def run_from_config(params, *args):
        if len(args) < 2:
            print "Invalid arguments!"
            sys.exit(-1)

        input_file = args[0]
        test_file = args[1]
        num_epochs = 1000
        if len(args) > 2:
            num_epochs = int(args[2])
        params["D"] = 2 ** params["logD"]
        params["conv_layer1_nfilters"] = 2 ** params["log_conv_layer1_nfilters"]
        params["conv_layer2_nfilters"] = 2 ** params["log_conv_layer2_nfilters"]

        model = ConvolutionalAttentionalLearner(params)
        model.train(input_file, max_epochs=num_epochs)

        test_data, original_names = model.naming_data.conv_data(test_file, model.name_cx_size, model.padding_size)
        test_name_targets, test_name_contexts, test_code_sentences, test_original_name_ids = test_data
        ids, unique_idx = np.unique(test_original_name_ids, return_index=True)

        eval = F1Evaluator(model)
        point_suggestion_eval = eval.compute_names(test_code_sentences[unique_idx], original_names,
                                                      model2.naming_data.all_tokens_dictionary.get_all_names())
        return -point_suggestion_eval.get_f1_at_all_ranks()[1]

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage <input_file> <max_num_epochs> d <test_file>'
        sys.exit(-1)

    input_file = sys.argv[1]
    max_num_epochs = int(sys.argv[2])
    params = {
        "D": int(sys.argv[3]),
        "name_cx_size": 1,
        "conv_layer1_nfilters": 64,
        "conv_layer2_nfilters": 16,
        "layer1_window_size": 6,
        "layer2_window_size": 15,
        "layer3_window_size": 14,
        "log_code_rep_init_scale": -1.34,
        "log_name_rep_init_scale": -4.9,
        "log_layer1_init_scale": -1,
        "log_layer2_init_scale": -3.4,
        "log_layer3_init_scale": -1.8,
        "log_name_cx_init_scale": -1.3,
        "log_learning_rate": -2.95,
        "rmsprop_rho": .98,
        "momentum": 0.9,
        "dropout_rate": 0.25,
        "grad_clip":1
    }

    params["train_file"] = input_file
    params["test_file"] = sys.argv[4]

    model = ConvolutionalAttentionalLearner(params)

    model.train(input_file, max_epochs=max_num_epochs)

    model.save("convolutional_att_model" + os.path.basename(params["train_file"]) + ".pkl")

    model2 = ConvolutionalAttentionalLearner.load("convolutional_att_model" + os.path.basename(params["train_file"]) + ".pkl")

    test_data, original_names = model2.naming_data.conv_data(sys.argv[4], model2.name_cx_size, model2.padding_size)
    test_name_targets, test_name_contexts, test_code_sentences, test_original_name_ids = test_data
    name_ll = model2.model.log_prob_with_targets(test_name_contexts, test_code_sentences, test_name_targets)
    print "Test name_ll=%s" % name_ll

    ids, unique_idx = np.unique(test_original_name_ids, return_index=True)
    eval = F1Evaluator(model2)
    point_suggestion_eval = eval.compute_names(test_code_sentences[unique_idx], original_names,
                                                  model2.naming_data.all_tokens_dictionary.get_all_names())
    print point_suggestion_eval
    results = point_suggestion_eval.get_f1_at_all_ranks()
    print results
