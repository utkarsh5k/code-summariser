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
