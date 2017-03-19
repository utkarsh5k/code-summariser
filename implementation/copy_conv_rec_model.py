import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from optimization import nesterov_rmsprop_multiple

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

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D",
                      "log_name_rep_init_scale",
                      "conv_layer1_nfilters", "layer1_window_size", "log_layer1_init_scale",
                      "conv_layer2_nfilters", "layer2_window_size", "log_layer2_init_scale",
                      "layer3_window_size", "log_layer3_init_scale",
                      "log_copy_init_scale", "log_hidden_init_scale",
                      "log_learning_rate", "momentum", "rmsprop_rho", "dropout_rate", "grad_clip"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def restore_parameters(self, values):
        for value, param in zip(values, self.train_parameters):
            param.set_value(value)
        self.__compile_model_functions()


    def __init_parameter(self, empirical_name_dist):
        all_name_rep = np.random.randn(self.all_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.all_name_reps = theano.shared(all_name_rep.astype(floatX), name="code_name_reps")

        # By convention, the last one is NONE, which is never predicted.
        self.name_bias = theano.shared(np.log(empirical_name_dist).astype(floatX)[:-1], name="name_bias")

        conv_layer1_code = np.random.randn(self.hyperparameters["conv_layer1_nfilters"], 1,
                                     self.hyperparameters["layer1_window_size"], self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_code = theano.shared(conv_layer1_code.astype(floatX), name="conv_layer1_code")
        conv_layer1_bias = np.random.randn(self.hyperparameters["conv_layer1_nfilters"]) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_bias = theano.shared(conv_layer1_bias.astype(floatX), name="conv_layer1_bias")

        # Comflate to 1D
        conv_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"], self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_code = theano.shared(conv_layer2_code.astype(floatX), name="conv_layer2_code")

        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias.astype(floatX), name="conv_layer2_bias")

        # Probability that each token will be copied
        conv_layer3_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_copy_code = theano.shared(conv_layer3_code.astype(floatX), name="conv_layer3_copy_code")
        conv_layer3_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_copy_bias = theano.shared(conv_layer3_bias[0].astype(floatX), name="conv_layer3_copy_bias")

        # Probability that we do a copy
        conv_copy_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.conv_copy_code = theano.shared(conv_copy_code.astype(floatX), name="conv_copy_code")

        conv_copy_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.conv_copy_bias = theano.shared(conv_copy_bias[0].astype(floatX), name="conv_copy_bias")

        # Attention vectors
        conv_layer3_att_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_att_code = theano.shared(conv_layer3_att_code.astype(floatX), name="conv_layer3_att_code")

        conv_layer3_att_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_att_bias = theano.shared(conv_layer3_att_bias[0].astype(floatX), name="conv_layer3_att_bias")

        # Recurrent layer
        gru_prev_hidden_to_next = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_next = theano.shared(gru_prev_hidden_to_next.astype(floatX), name="gru_prev_hidden_to_next")
        gru_prev_hidden_to_reset = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_reset = theano.shared(gru_prev_hidden_to_reset.astype(floatX), name="gru_prev_hidden_to_reset")
        gru_prev_hidden_to_update = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_update = theano.shared(gru_prev_hidden_to_update.astype(floatX), name="gru_prev_hidden_to_update")

        gru_prediction_to_reset = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_reset = theano.shared(gru_prediction_to_reset.astype(floatX), name="gru_prediction_to_reset")

        gru_prediction_to_update = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_update = theano.shared(gru_prediction_to_update.astype(floatX), name="gru_prediction_to_update")

        gru_prediction_to_hidden = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_hidden = theano.shared(gru_prediction_to_hidden.astype(floatX), name="gru_prediction_to_hidden")

        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias.astype(floatX), name="conv_layer2_bias")

        gru_hidden_update_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_hidden_update_bias = theano.shared(gru_hidden_update_bias.astype(floatX), name="gru_hidden_update_bias")
        gru_update_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_update_bias = theano.shared(gru_update_bias.astype(floatX), name="gru_update_bias")
        gru_reset_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_reset_bias = theano.shared(gru_reset_bias.astype(floatX), name="gru_reset_bias")

        h0 = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.h0 = theano.shared(h0.astype(floatX), name="h0")

        self.rng = RandomStreams()
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3

        self.train_parameters = [self.all_name_reps,
                                 self.conv_layer1_code, self.conv_layer1_bias,
                                 self.conv_layer2_code, self.conv_layer2_bias,
                                 self.conv_layer3_copy_code, self.conv_layer3_copy_bias,
                                 self.conv_copy_code, self.conv_copy_bias,self.h0,
                                 self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update,
                                 self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update,
                                 self.gru_hidden_update_bias, self.gru_update_bias, self.gru_reset_bias,
                                 self.conv_layer3_att_code, self.conv_layer3_att_bias, self.name_bias]

        self.__compile_model_functions()

    def __compile_model_functions(self):
        grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(floatX)) for param in self.train_parameters] \
                    + [theano.shared(np.zeros(1,dtype=floatX)[0], name="sentence_count")]

        sentence = T.ivector("sentence")
        is_copy_matrix = T.imatrix("is_copy_matrix")
        name_targets = T.ivector("name_targets")
        targets_is_unk = T.ivector("targets_is_unk")

        #theano.config.compute_test_value = 'warn'
        sentence.tag.test_value = np.arange(105).astype(np.int32)
        name_targets.tag.test_value = np.arange(5).astype(np.int32)
        targets_is_unk.tag.test_value = np.array([0, 0, 1, 0, 0], dtype=np.int32)
        is_copy_test_value = [[i % k == k-2 for i in xrange(105 - self.padding_size)] for k in [1, 7, 10, 25, 1]]
        is_copy_matrix.tag.test_value = np.array(is_copy_test_value, dtype=np.int32)

        _, _, copy_weights, _, copy_probs, name_log_probs, _\
                = self.__get_model_likelihood_for_sentence(sentence, name_targets, do_dropout=True, dropout_rate=self.hyperparameters["dropout_rate"])

        correct_answer_log_prob = self.model_objective(copy_probs[:-1], copy_weights[:-1], is_copy_matrix[1:], name_log_probs[:-1], name_targets[1:], targets_is_unk[1:])

        grad = T.grad(correct_answer_log_prob, self.train_parameters)
        self.grad_accumulate = theano.function(inputs=[sentence, is_copy_matrix, targets_is_unk, name_targets], updates=[(v, v+g) for v, g in zip(grad_acc, grad)] + [(grad_acc[-1], grad_acc[-1]+1)], #mode='NanGuardMode')

        normalized_grads = [T.switch(grad_acc[-1] >0, g / grad_acc[-1], g) for g in grad_acc[:-1]]
        step_updates, ratios = nesterov_rmsprop_multiple(self.train_parameters, normalized_grads,
                                                learning_rate=10 ** self.hyperparameters["log_learning_rate"],
                                                rho=self.hyperparameters["rmsprop_rho"],
                                                momentum=self.hyperparameters["momentum"],
                                                grad_clip=self.hyperparameters["grad_clip"],
                                                output_ratios=True)
        step_updates.extend([(v, T.zeros(v.shape)) for v in grad_acc[:-1]])  # Set accumulators to 0
        step_updates.append((grad_acc[-1], 0))

        self.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)


        test_sentence, test_name_targets, test_copy_weights, test_attention_weights, test_copy_probs, test_name_log_probs,\
                         test_attention_features \
            = self.__get_model_likelihood_for_sentence(T.ivector("test_sentence"),  T.ivector("test_name_targets"), do_dropout=False)

        self.copy_probs = theano.function(inputs=[test_name_targets, test_sentence], outputs=[test_copy_weights, test_copy_probs, test_name_log_probs])

        test_copy_matrix = T.imatrix("test_copy_matrix")
        test_target_is_unk = T.ivector("test_target_is_unk")
        ll = self.model_objective(test_copy_probs[:-1], test_copy_weights[:-1], test_copy_matrix[1:], test_name_log_probs[:-1], test_name_targets[1:], test_target_is_unk[1:])
        self.copy_logprob = theano.function(inputs=[test_sentence, test_copy_matrix, test_target_is_unk, test_name_targets], outputs=ll)
        self.attention_weights = theano.function(inputs=[test_name_targets, test_sentence], outputs=test_attention_weights)
        layer3_padding = self.hyperparameters["layer3_window_size"] - 1
        upper_pos = -layer3_padding/2+1 if -layer3_padding/2+1 < 0 else None
        self.attention_features = theano.function(inputs=[test_sentence, test_name_targets], outputs=test_attention_features[-1, 0, :, layer3_padding/2+1:upper_pos, 0])

    


