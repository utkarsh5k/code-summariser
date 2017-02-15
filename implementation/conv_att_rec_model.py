import theano
from theano import tensor as T
import numpy as np

floatX = theano.config.floatX

class ConvolutionalAttentionalRecurrentModel(object):
    
    def __init__(self, hyperparameters, all_voc_size, empirical_name_dist):
        self.D = hyperparameters["D"]

        self.hyperparameters = hyperparameters

        self.all_voc_size = all_voc_size

    def log_prob_with_targets(self, sentence, name_targets):
        ll = 0
        for i in xrange(len(name_targets)):
            ll += self.__log_prob_with_targets(sentence[i], name_targets[i])
        return (ll / len(name_targets))

    def log_prob(self, name_contexts, sentence):
        ll = []
        for i in xrange(len(sentence)):
            log_probs = self.__log_prob_last(sentence[i], name_contexts[i])
            ll.append(log_probs)
        return np.array(ll)