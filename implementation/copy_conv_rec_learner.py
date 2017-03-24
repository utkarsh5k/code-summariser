import cPickle
import sys
import os
import time
import re

import numpy as np

from collections import defaultdict
from math import ceil
from experimenter import ExperimentLogger

from copy_conv_rec_model import CopyConvolutionalRecurrentAttentionalModel
from formatting_tokens import FormatTokens
from f1_score import F1Evaluator

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

def run_from_config(params, *args):
    if len(args) < 2:
        print "No input file or test file given: %s:%s" % (args, len(args))
        sys.exit(-1)
    input_file = args[0]
    test_file = args[1]
    if len(args) > 2:
        num_epochs = int(args[2])
    else:
        num_epochs = 1000

    params["D"] = 2 ** params["logD"]
    params["conv_layer1_nfilters"] = 2 ** params["log_conv_layer1_nfilters"]
    params["conv_layer2_nfilters"] = 2 ** params["log_conv_layer2_nfilters"]

    model = ConvolutionalCopyAttentionalRecurrentLearner(params)
    model.train(input_file, max_epochs=num_epochs)

    test_data, original_names = model.naming_data.data_in_rec_copy_conv_format(test_file, model.padding_size)
    test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data
    eval = F1Evaluator(model)
    point_suggestion_eval = eval.compute_names_f1(test_code, original_names, model.naming_data.all_tokens_dictionary.get_all_names())
    return -point_suggestion_eval.get_f1_at_all_ranks()[1]

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print 'Usage <input_file> <max_num_epochs> d <test_file>'
        sys.exit(-1)

    input_file = sys.argv[1]
    max_num_epochs = int(sys.argv[2])
    params = {
        "D": int(sys.argv[3]),
        "conv_layer1_nfilters": 32,
        "conv_layer2_nfilters": 16,
        "layer1_window_size": 18,
        "layer2_window_size": 19,
        "layer3_window_size": 2,
        "log_name_rep_init_scale": -1,
        "log_layer1_init_scale": -3.68,
        "log_layer2_init_scale": -4,
        "log_layer3_init_scale": -4,
        "log_hidden_init_scale": -1,
        "log_copy_init_scale":-0.5,
        "log_learning_rate": -3.05,
        "rmsprop_rho": .99,
        "momentum": 0.87,
        "dropout_rate": 0.4,
        "grad_clip":.75
    }

    params["train_file"] = input_file
    if len(sys.argv) > 4:
        params["test_file"] = sys.argv[4]
    with ExperimentLogger("ConvolutionalCopyAttentionalRecurrentLearner", params) as experiment_log:
        if max_num_epochs:
            model = ConvolutionalCopyAttentionalRecurrentLearner(params)
            model.train(input_file, max_epochs=max_num_epochs)
            model.save("copy_convolutional_att_rec_model" + os.path.basename(params["train_file"]) + ".pkl")

        if params.get("test_file") is None:
            exit()

        model2 = ConvolutionalCopyAttentionalRecurrentLearner.load("copy_convolutional_att_rec_model" + os.path.basename(params["train_file"]) + ".pkl")

        test_data, original_names = model2.naming_data.data_in_rec_copy_conv_format(sys.argv[4], model2.padding_size)
        test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data

        eval = F1Evaluator(model2)
        point_suggestion_eval = eval.compute_names(test_code, original_names, model2.naming_data.all_tokens_dictionary.get_all_names())
        print point_suggestion_eval
        results = point_suggestion_eval.get_f1_at_all_ranks()
        print results
        experiment_log.record_results({"f1_at_rank1": results[0], "f1_at_rank5":results[1]})


    