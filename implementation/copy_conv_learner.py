import os
import sys
import re
import time
import cPickle

from collections import defaultdict
from math import ceil

import numpy as np

from copy_conv_model import CopyConvolutionalAttentionalModel
from formatting_tokens import FormatTokens
from f1_score import F1Evaluator

class CopyAttentionalLearner:
	def __init__(self, hyperparameters):
		self.name_cx_size = hyperparameters["name_cx_size"]
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def train(self, inp, patience=5, max_epochs=1000, minibatch_size=500):
        assert self.parameters is None, "Already trained model"
        print "Data extraction ongoing"
        train_data, validation_data, self.naming_data = FormatTokens.validated_copy_conv_data(inp, self.name_cx_size, .92, self.padding_size)
        train_name_targets, train_original_targets, train_name_contexts, train_code_sentences, train_code, train_copy_vectors, train_target_is_unk, train_original_name_ids = train_data
        val_name_targets, val_original_targets, val_name_contexts, val_code_sentences, val_code, val_copy_vectors, val_target_is_unk, val_original_name_ids = validation_data
        model = CopyConvolutionalAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary), len(self.naming_data.name_dictionary), self.naming_data.name_empirical_dist)

        def log_probablity_for_validation():
            return model.log_prob_no_predict(val_name_contexts, val_code_sentences, val_copy_vectors, val_target_is_unk, val_name_targets)

        best_params = [p.get_value() for p in model.train_parameters]
        best_name_score = float('-inf')
        ratios = np.zeros(len(model.train_parameters))
        n_batches = 0
        epochs_not_improved = 0
        print "Training start at [%s]" % time.asctime()
        for i in xrange(max_epochs):
            start_time = time.time()
            name_ordering = np.arange(len(train_name_targets), dtype=np.int32)
            np.random.shuffle(name_ordering)
            sys.stdout.write(str(i))
            num_minibatches = min(int(ceil(float(len(train_name_targets)) / minibatch_size))-1, 25)

            for j in xrange(num_minibatches):
                name_batch_ids = name_ordering[j * minibatch_size:(j + 1) * minibatch_size]
                batch_code_sentences = train_code_sentences[name_batch_ids]
                for k in xrange(len(name_batch_ids)):
                        idx = name_batch_ids[k]
                        if train_target_is_unk[idx] == 1 and np.sum(train_copy_vectors[idx]) == 0:
                            continue
                        model.grad_accumulate(train_name_contexts[idx], batch_code_sentences[k],
                                              train_copy_vectors[idx], train_target_is_unk[idx],
                                              train_name_targets[idx])
                assert len(name_batch_ids) > 0
                ratios += model.grad_step()
                n_batches += 1
            sys.stdout.write("|")
            if i % 1 == 0:
                name_ll = compute_validation_logprob()
                if name_ll > best_name_score:
                    best_name_score = name_ll
                    best_params = [p.get_value() for p in model.train_parameters]
                    print "At %s validation: name_ll=%s [best so far]" % (i, name_ll)
                    epochs_not_improved = 0
                else:
                    print "At %s validation: name_ll=%s" % (i, name_ll)
                    epochs_not_improved += 1
                for k in xrange(len(model.train_parameters)):
                    print "%s: %.0e" % (model.train_parameters[k].name, ratios[k] / n_batches)
                n_batches = 0
                ratios = np.zeros(len(model.train_parameters))
            if epochs_not_improved >= patience:
                print "Not improved for %s epochs. Stopping" % patience
                break
            elapsed = int(time.time() - start_time)
            print "Epoch elapsed %sh%sm%ss" % ((elapsed / 60 / 60) % 60, (elapsed / 60) % 60, elapsed % 60)
        print "Training Finished at [%s]" % time.asctime()
        self.parameters = best_params
        model.restore_parameters(best_params)
        self.model = model

    def save(self, filename):
        model_tmp = self.model
        del self.model
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        self.model = model_tmp

    def load(filename):
        """
        :type filename: str
        :rtype: CopyAttentionalLearner
        """
        with open(filename, 'rb') as f:
            learner = cPickle.load(f)
        learner.model = CopyConvolutionalAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary), len(learner.naming_data.name_dictionary), learner.naming_data.name_empirical_dist)
        learner.model.restore_parameters(learner.parameters)
        return learner