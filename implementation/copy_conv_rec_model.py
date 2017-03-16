import theano
from theano import tensor as T
import numpy as np

floatX = theano.config.floatX

class CopyConvolutionalRecurrentAttentionalModel(object):
    def __init__(self, hyperparameters, all_voc_size, empirical_name_dist):
        self.D = hyperparameters["D"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size

        self.__init_parameter(empirical_name_dist)

    def log_prob_no_predict(self, name_contexts, sentences, copy_vectors, target_is_unk, name_target):
        ll = 0
        for i in xrange(len(sentences)):
            ll += self.copy_logprob(name_contexts[i], sentences[i], copy_vectors[i], target_is_unk[i], name_target[i])
        return (ll / len(sentences))

    def log_prob_with_targets(self, sentence, copy_matrices, targets_is_unk, name_targets):
        ll = 0
        for i in xrange(len(name_targets)):
            ll += self.copy_logprob(sentence[i], copy_matrices[i], targets_is_unk[i], name_targets[i])
        return (ll / len(name_targets))