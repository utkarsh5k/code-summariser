import theano
from theano import tensor as T
import numpy as np

floatX = theano.config.floatX

class ConvolutionalAttentionalModel(object):

    def __init__(self, hyperparameters, all_voc_size, name_voc_size, empirical_name_dist):
        self.D = hyperparameters["D"]
        self.name_cx_size = hyperparameters["name_cx_size"]
        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size
        self.name_voc_size = name_voc_size
        self.__init_parameter(empirical_name_dist)

    def __init_parameter(self, empirical_name_dist):
        all_name_rep = np.random.randn(self.all_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.all_name_reps = theano.shared(all_name_rep.astype(floatX), name="all_name_reps")

        name_cx_rep = np.random.randn(self.name_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.name_cx_reps = theano.shared(name_cx_rep.astype(floatX), name="name_cx_rep")

        conv_layer1_code = np.random.randn(self.hyperparameters["conv_layer1_nfilters"], 1,
                                     self.hyperparameters["layer1_window_size"], self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_code = theano.shared(conv_layer1_code.astype(floatX), name="conv_layer1_code")
        conv_layer1_bias = np.random.randn(self.hyperparameters["conv_layer1_nfilters"]) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_bias = theano.shared(conv_layer1_bias.astype(floatX), name="conv_layer1_bias")

        conv_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"],
                                     self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_code = theano.shared(conv_layer2_code.astype(floatX), name="conv_layer2_code")
        gate_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"],
                                     self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gate_weights_code_l2 = theano.shared(gate_layer2_code.astype(floatX), name="gate_weights_code_l2")
        conv_layer2_name_cx = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"], self.name_cx_size, self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer2_name_cx = theano.shared(conv_layer2_name_cx.astype(floatX), name="conv_layer2_name_cx")

        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias.astype(floatX), name="conv_layer2_bias")
        gate_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gate_layer2_bias = theano.shared(gate_layer2_bias.astype(floatX), name="gate_layer2_bias")

        conv_layer3_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_code = theano.shared(conv_layer3_code.astype(floatX), name="conv_layer3_code")
        conv_layer3_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_bias = theano.shared(conv_layer3_bias[0].astype(floatX), name="conv_layer3_bias")

        self.name_bias = theano.shared(np.log(empirical_name_dist).astype(floatX)[:-1], name="name_bias")

        self.rng = RandomStreams()

        self.train_parameters = [self.all_name_reps, self.name_cx_reps,
                                 self.conv_layer1_code, self.conv_layer2_name_cx, self.conv_layer1_bias,
                                 self.conv_layer2_code, self.conv_layer2_bias, self.conv_layer3_code, self.conv_layer3_bias,
                                 self.name_bias, self.gate_layer2_bias, self.gate_weights_code_l2]

        self.__compile_model_functions()

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D", "name_cx_size",
                      "log_code_rep_init_scale", "log_name_rep_init_scale",
                      "conv_layer1_nfilters", "layer1_window_size", "log_layer1_init_scale",
                      "conv_layer2_nfilters", "layer2_window_size", "log_layer2_init_scale",
                      "layer3_window_size", "log_layer3_init_scale",
                      "log_name_cx_init_scale",
                      "log_learning_rate", "momentum", "rmsprop_rho", "dropout_rate", "grad_clip"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def restore_parameters(self, values):
        for value, param in zip(values, self.train_parameters):
            param.set_value(value)
        self.__compile_model_functions()




    def log_prob_with_targets(self, name_contexts, sentence, name_targets):
        ll = 0
        for i in xrange(len(name_targets)):
            ll += self.__log_prob_with_targets(name_contexts[i], sentence[i].astype(np.int32), name_targets[i])
        return (ll / len(name_targets))

    def log_prob(self, name_contexts, sentence):
        ll = []
        for i in xrange(len(sentence)):
            log_probs = self.__log_prob(name_contexts[i], sentence[i])[0]
            ll.append(log_probs)
        return np.array(ll)
