import cPickle
import sys
import os
import time

from collections import defaultdict

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

    