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