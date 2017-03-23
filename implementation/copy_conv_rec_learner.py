import cPickle
import sys
import os
import time
import re

import numpy as np

from collections import defaultdict
from math import ceil

from copy_conv_rec_model import CopyConvolutionalRecurrentAttentionalModel
from formatting_tokens import FormatTokens

class ConvolutionalCopyAttentionalRecurrentLearner:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def save(self, filename):
        model_tmp = self.model
        del self.model
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        self.model = model_tmp

    def load(filename):
        with open(filename, 'rb') as f:
            learner = cPickle.load(f)
        learner.model = CopyConvolutionalRecurrentAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary), learner.naming_data.name_empirical_dist)
        learner.model.restore_parameters(learner.parameters)
        return learner

    def train(self, input_file, patience=5, max_epochs=1000, minibatch_size=500):
        assert self.parameters is None, "Already trained model"
        print "saving best result %s"%("copy_convolutional_att_rec_model" + os.path.basename(self.hyperparameters["train_file"]) + ".pkl")
        print "Extracting data"

        train_data, validation_data, self.naming_data = FormatTokens.rec_copy_conv_data(input_file, .92, self.padding_size)
        train_name_targets, train_code_sentences, train_code, train_target_is_unk, train_copy_vectors = train_data
        val_name_targets, val_code_sentences, val_code, val_target_is_unk, val_copy_vectors = validation_data

        model = CopyConvolutionalRecurrentAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary), self.naming_data.name_empirical_dist)
        self.model = model

        def compute_validation_score_names():
            return model.log_prob_with_targets(val_code_sentences, val_copy_vectors, val_target_is_unk, val_name_targets)

        best_params = [p.get_value() for p in model.train_parameters]
        best_name_score = float('-inf')
        ratios = np.zeros(len(model.train_parameters))
        n_batches = 0
        epochs_not_improved = 0

        print "[%s] Train start" % time.asctime()
        for i in xrange(max_epochs):
            start_time = time.time()
            name_ordering = np.arange(len(train_name_targets), dtype=np.int32)
            np.random.shuffle(name_ordering)

            sys.stdout.write(str(i))
            num_minibatches = min(int(ceil(float(len(train_name_targets)) / minibatch_size))-1, 25)
            for j in xrange(num_minibatches):
                if (j + 1) * minibatch_size > len(name_ordering):
                    j = 0
                name_batch_ids = name_ordering[j * minibatch_size:(j + 1) * minibatch_size]
                batch_code_sentences = train_code_sentences[name_batch_ids]
                for k in xrange(len(name_batch_ids)):
                    pos = name_batch_ids[k]
                    model.grad_accumulate(batch_code_sentences[k], train_copy_vectors[pos], train_target_is_unk[pos], train_name_targets[pos])
                assert len(name_batch_ids) > 0
                ratios += model.grad_step()
                sys.stdout.write("\r%d %d"%(i, n_batches))
                n_batches += 1
            sys.stdout.write("|")
            if i % 1 == 0:
                name_ll = compute_validation_score_names()
                if name_ll > best_name_score:
                    best_name_score = name_ll
                    best_params = [p.get_value() for p in model.train_parameters]
                    self.parameters = best_params
                    print "At %s validation: name_ll=%s [best so far]" % (i, name_ll)
                    epochs_not_improved = 0
                    self.save("copy_convolutional_att_rec_model" + os.path.basename(self.hyperparameters["train_file"]) + ".pkl")
                else:
                    print "At %s validation: name_ll=%s" % (i, name_ll)
                    epochs_not_improved += 1
                for k in xrange(len(model.train_parameters)):
                    print "%s: %.0e" % (model.train_parameters[k].name, ratios[k] / n_batches)
                n_batches = 0
                ratios = np.zeros(len(model.train_parameters))

            if epochs_not_improved >= patience:
                print "Not improved for %s epochs. Stop training." % patience
                break
            elapsed = int(time.time() - start_time)
            print "Epoch elapsed %sh%sm%ss" % ((elapsed / 60 / 60) % 60, (elapsed / 60) % 60, elapsed % 60)
        print "[%s] Training Over" % time.asctime()
        self.parameters = best_params
        model.restore_parameters(best_params)

    identifier_matcher = re.compile('[a-zA-Z0-9]+')

    