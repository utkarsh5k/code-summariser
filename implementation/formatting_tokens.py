from feature_set import feature_set
import json
from itertools import chain, repeat
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import heapq

"""
most data formatting functions have lists being converted to numpy arrays since everything useful has
to be converted to np objects for further use

validation inclusive methods follow the same template for all data formats be it conv, rec, copy conv,
or rec copy conv
"""

class FormatTokens:

	SUBTOKEN_START = "%START%"
    SUBTOKEN_END = "%END%"
    NONE = "%NONE%"

    def __init__(self, names, code):
        self.name_dictionary = FeatureSet.make_feature_set(chain.from_iterable(names), 2)
        self.name_dictionary.id_from_token(self.NONE)
        self.all_tokens_dictionary = FeatureSet.make_feature_set(chain.from_iterable([chain.from_iterable(code), chain.from_iterable(names)]), 5)
        self.all_tokens_dictionary.id_from_token(self.NONE)
        self.name_empirical_dist = self.__create_empirical_distribution(self.all_tokens_dictionary, chain.from_iterable(names))

    def __read_file(inp):
        with open(inp, 'r') as f:
            data = json.load(f)
        names = []
        original_names = []
        code = []
        for datum in data:
            if len(datum["tokens"]) == 0 or len(datum["name"]) == 0:
            	"""
            	skip over records with no relevant data, so less computation involved
            	"""
                continue
            code.append(FormatTokens.remove_id_tags(datum["tokens"]))
            original_names.append(",".join(datum["name"]))
            subtokens = datum["name"]
            names.append([FormatTokens.SUBTOKEN_START] + subtokens + [FormatTokens.SUBTOKEN_END])

        return names, code, original_names

    def remove_id_tags(code):
        return filter(lambda x: x != "<id>" and x != "</id>", code)

    def id_tags_only(self, code):
        tags_code = []
        for tokens in code:
            id_tokens = []
            in_id = False
            for subtoken in tokens:
                if subtoken == "<id>":
                    in_id = True
                elif subtoken == '</id>':
                    in_id = False
                elif in_id:
                    id_tokens.append(subtoken)
            tags_code.append(id_tokens)
        return tags_code

    """
    scaling parameter to be used for dirichlet stochastic process, value determined by previous work
    """
    def __create_empirical_distribution(records_dict, records, alpha_dirichlet=10.):
    	targets = np.array([records_dict.is_id_or_is_unknown(token) for token in record])
    	"""
    	number of occurences of each value in non-negative integer array
    	"""
        empirical_dist = np.bincount(targets, minlength=len(records_dict)).astype(float)
        empirical_dist += alpha_dirichlet / len(empirical_dist)
        return empirical_dist / (np.sum(empirical_dist) + alpha_dirichlet)

    """
    took too long to figure out, clearly help taken
    creating targets, context and ids for data
    """
    def __label_format(self, data, dictionary, cx_size):
        targets = []
        contexts = []
        ids = []
        for i, sequence in enumerate(data):
            for j in xrange(1, len(sequence)): # First element always predictable
                ids.append(i)
                targets.append(dictionary.is_id_or_is_unknown(sequence[j]))
                context = sequence[:j]
                if len(context) < cx_size:
                    context = [self.NONE] * (cx_size - len(context)) + context
                else:
                    context = context[-cx_size:]
                assert len(context) == cx_size, (len(context), cx_size,)
                contexts.append([dictionary.is_id_or_is_unknown(c) for c in context])
        return np.array(targets, dtype=np.int32), np.array(contexts, dtype=np.int32), np.array(ids, np.int32)

    """
    return labeled tokens, labeled names and original names
    """
    def get_label_data(self, inp, code_cx_size, names_cx_size):
        names, code, original_names = self.__read_file(inp)
        return self.__label_format(names, self.name_dictionary, names_cx_size), self.__label_format(code, self.all_tokens_dictionary, code_cx_size), original_names

    def validated_label_data(inp, code_cx_size, names_cx_size, percent_train):
        """
        percentages cannot be less than 0 or greater than 1, can't believe this needs a check
        """
        assert percent_train < 1
        assert percent_train > 0
        names, code, original_names = FormatTokens.__read_file(inp)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        original_names = np.array(original_names, dtype=np.object)
        limit = int(percent_train * len(names))
        naming = FormatTokens(names[:limit], code[:limit])
        """
        explicit line joining necessary here
        """
        return naming.__label_format(names[:limit], naming.name_dictionary, names_cx_size), \
               naming.__label_format(code[:limit], naming.all_tokens_dictionary, code_cx_size), original_names[:limit], \
               naming.__label_format(names[limit:], naming.name_dictionary, names_cx_size), \
               naming.__label_format(code[limit:], naming.all_tokens_dictionary, code_cx_size), original_names[limit:], naming

    def forward_formatted_data(self, inp, name_cx_size):
        names, code, original_names = self.__read_file(inp)
        return self.__get_data_in_forward_format(names, code, name_cx_size), original_names

    def __forward_model_data(self, names, code, name_cx_size):
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        name_contexts = []
        original_names_ids = []
        id_xs = []
        id_ys = []
        k = 0
        for i, name in enumerate(names):
            for j in xrange(1, len(name)):  # First element always predictable
                name_targets.append(self.name_dictionary.is_id_or_is_unknown(name[j]))
                original_names_ids.append(i)
                context = name[:j]
                if len(context) < name_cx_size:
                    context = [self.NONE] * (name_cx_size - len(context)) + context
                else:
                    context = context[-name_cx_size:]
                assert len(context) == name_cx_size, (len(context), name_cx_size,)
                name_contexts.append([self.name_dictionary.is_id_or_is_unknown(token) for token in context])
                for code_token in set(code[i]):
                    token_id = self.all_tokens_dictionary.is_id_or_is_none(code_token)
                    if token_id is not None:
                        id_xs.append(k)
                        id_ys.append(token_id)
                k += 1
        code_features = sp.csr_matrix((np.ones(len(id_xs)), (id_xs, id_ys)), shape=(k, len(self.all_tokens_dictionary)), dtype=np.int32)
        name_targets = np.array(name_targets, dtype=np.int32)
        name_contexts = np.array(name_contexts, dtype=np.int32)
        original_names_ids = np.array(original_names_ids, dtype=np.int32)
        return name_targets, name_contexts, code_features, original_names_ids

    def validated_forward_format_data(inp, names_cx_size, percent_train):
        """
        percentages cannot be less than 0 or greater than 1, can't believe this needs a check
        """
        assert percent_train < 1
        assert percent_train > 0
        names, code, original_names = FormatTokens.__read_file(inp)
        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        original_names = np.array(original_names, dtype=np.object)
        lim = int(percent_train * len(names))
        naming = FormatTokens(names[:lim], code[:lim])
        return naming.__forward_model_data(names[:lim], code[:lim], names_cx_size),\
                naming.__forward_model_data(names[lim:], code[lim:], names_cx_size), naming

    def conv_data(self, names, code, name_cx_size, sentence_padding):
        """
        data regularity and shape check
        """
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        name_contexts = []
        original_names_ids = []
        sentences = []
        padding = [self.all_tokens_dictionary.is_id_or_is_unknown(self.NONE)]

        for i, name in enumerate(names):
            sentence = [self.all_tokens_dictionary.is_id_or_is_unknown(token) for token in code[i]]
            if sentence_padding % 2 == 0:
                sentence = padding * (sentence_padding / 2) + sentence + padding * (sentence_padding / 2)
            else:
                sentence = padding * (sentence_padding / 2 + 1) + sentence + padding * (sentence_padding / 2)
            for j in xrange(1, len(name)):  # First element always predictable
                name_targets.append(self.all_tokens_dictionary.is_id_or_is_unknown(name[j]))
                original_names_ids.append(i)
                context = name[:j]
                if len(context) < name_cx_size:
                    context = [self.NONE] * (name_cx_size - len(context)) + context
                else:
                    context = context[-name_cx_size:]
                assert len(context) == name_cx_size, (len(context), name_cx_size,)
                name_contexts.append([self.name_dictionary.is_id_or_is_unknown(token) for token in context])
                sentences.append(np.array(sentence, dtype=np.int32))

        name_targets = np.array(name_targets, dtype=np.int32)
        name_contexts = np.array(name_contexts, dtype=np.int32)
        sentences = np.array(sentences, dtype=np.object)
        original_names_ids = np.array(original_names_ids, dtype=np.int32)
        return name_targets, name_contexts, sentences, original_names_ids

    def data_in_conv_format(self, inp, name_cx_size, min_code_size):
        names, code, original_names = self.__read_file(inp)
        return self.conv_data(names, code, name_cx_size, min_code_size), original_names

    """
    similar to the forward format data with validation method
    """
    def validated_conv_data(inp, names_cx_size, percent_train, min_code_size):
        assert percent_train < 1
        assert percent_train > 0
        names, code, original_names = FormatTokens.__read_file(inp)
        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(percent_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = FormatTokens(names[idxs[:lim]], code[idxs[:lim]])
        return naming.conv_data(names[idxs[:lim]], code[idxs[:lim]], names_cx_size, min_code_size),\
                naming.conv_data(names[idxs[lim:]], code[idxs[lim:]], names_cx_size, min_code_size), naming

    def data_in_rec_conv_format(self, inp, min_code_size):
        names, code, original_names = self.__read_file(inp)
        return self.rec_conv_data(names, code, min_code_size), original_names

    """
    looks similar to convolution data formatting, but it really isn't
    """
    def rec_conv_data(self, names, code, sentence_padding):
        """
        data regularity and shape check
        """
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        sentences = []
        padding = [self.all_tokens_dictionary.is_id_or_is_unknown(self.NONE)]
        for i, name in enumerate(names):
            sentence = [self.all_tokens_dictionary.is_id_or_is_unknown(token) for token in code[i]]
            if sentence_padding % 2 == 0:
                sentence = padding * (sentence_padding / 2) + sentence + padding * (sentence_padding / 2)
            else:
                sentence = padding * (sentence_padding / 2 + 1) + sentence + padding * (sentence_padding / 2)
            name_tokens = [self.all_tokens_dictionary.is_id_or_is_unknown(token) for token in name]
            name_targets.append(np.array(name_tokens, dtype=np.int32))
            sentences.append(np.array(sentence, dtype=np.int32))
        name_targets = np.array(name_targets, dtype=np.object)
        sentences = np.array(sentences, dtype=np.object)
        return name_targets, sentences

    """
    this is similar to any of the previous validation methods written, for forward format and for
    convolution format
    """
    def validated_rec_conv_data(inp, percent_train, min_code_size):
        assert percent_train < 1
        assert percent_train > 0
        names, code, original_names = FormatTokens.__read_file(inp)
        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(percent_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = FormatTokens(names[idxs[:lim]], code[idxs[:lim]])
        return naming.rec_conv_data(names[idxs[:lim]], code[idxs[:lim]], min_code_size),\
                naming.rec_conv_data(names[idxs[lim:]], code[idxs[lim:]], min_code_size), naming

    def data_in_copy_conv_format(self, inp, name_cx_size, min_code_size):
        names, code, original_names = self.__read_file(inp)
        return self.copy_conv_data(names, code, name_cx_size, min_code_size), original_names

    def copy_conv_data(self, names, code, name_cx_size, sentence_padding):
        """
        data regularity and shape check
        """
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        original_targets = []
        name_contexts = []
        original_names_ids = []
        sentences = []
        original_code = []
        copy_vector = []
        target_is_unknown = []
        padding = [self.all_tokens_dictionary.is_id_or_is_unknown(self.NONE)]

        for i, name in enumerate(names):
            sentence = [self.all_tokens_dictionary.is_id_or_is_unknown(token) for token in code[i]]
            if sentence_padding % 2 == 0:
                sentence = padding * (sentence_padding / 2) + sentence + padding * (sentence_padding / 2)
            else:
                sentence = padding * (sentence_padding / 2 + 1) + sentence + padding * (sentence_padding / 2)
            for j in xrange(1, len(name)):  # First element always predictable like in convolution format
                name_targets.append(self.all_tokens_dictionary.is_id_or_is_unknown(name[j]))
                original_targets.append(name[j])
                target_is_unknown.append(self.all_tokens_dictionary.is_unknown(name[j]))
                original_names_ids.append(i)
                context = name[:j]
                if len(context) < name_cx_size:
                    context = [self.NONE] * (name_cx_size - len(context)) + context
                else:
                    context = context[-name_cx_size:]
                assert len(context) == name_cx_size, (len(context), name_cx_size,)
                name_contexts.append([self.name_dictionary.is_id_or_is_unknown(token) for token in context])
                sentences.append(np.array(sentence, dtype=np.int32))
                original_code.append(code[i])
                tokens_to_be_copied = [token == name[j] for token in code[i]]
                copy_vector.append(np.array(tokens_to_be_copied, dtype=np.int32))
        name_targets = np.array(name_targets, dtype=np.int32)
        name_contexts = np.array(name_contexts, dtype=np.int32)
        sentences = np.array(sentences, dtype=np.object)
        original_names_ids = np.array(original_names_ids, dtype=np.int32)
        copy_vector = np.array(copy_vector, dtype=np.object)
        target_is_unknown = np.array(target_is_unknown, dtype=np.int32)
        return name_targets, original_targets, name_contexts, sentences, original_code, copy_vector, target_is_unknown, original_names_ids

    def validated_copy_conv_data(inp, names_cx_size, percent_train, min_code_size):
        assert percent_train < 1
        assert percent_train > 0
        names, code, original_names = FormatTokens.__read_file(inp)
        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(percent_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = FormatTokens(names[idxs[:lim]], code[idxs[:lim]])
        return naming.copy_conv_data(names[idxs[:lim]], code[idxs[:lim]], names_cx_size, min_code_size),\
                naming.copy_conv_data(names[idxs[lim:]], code[idxs[lim:]], names_cx_size, min_code_size), naming

    def data_in_rec_copy_conv_format(self, inp, min_code_size):
        names, code, original_names = self.__get_file_data(inp)
        return self.rec_copy_conv_data(names, code, min_code_size), original_names

    def rec_copy_conv_data(self, names, code, sentence_padding):
        """
        data regularity and shape check
        """
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        target_is_unknown = []
        copy_vectors = []
        sentences = []
        padding = [self.all_tokens_dictionary.is_id_or_is_unknown(self.NONE)]
        for i, name in enumerate(names):
            sentence = [self.all_tokens_dictionary.is_id_or_is_unknown(token) for token in code[i]]
            if sentence_padding % 2 == 0:
                sentence = padding * (sentence_padding / 2) + sentence + padding * (sentence_padding / 2)
            else:
                sentence = padding * (sentence_padding / 2 + 1) + sentence + padding * (sentence_padding / 2)
            name_tokens = [self.all_tokens_dictionary.is_id_or_is_unknown(token) for token in name]
            unknown_tokens = [self.all_tokens_dictionary.is_unknown(token) for token in name]
            copiable_targets = [[token == subtoken for token in code[i]] for subtoken in name]
            name_targets.append(np.array(name_tokens, dtype=np.int32))
            target_is_unknown.append(np.array(unknown_tokens, dtype=np.int32))
            copy_vectors.append(np.array(copiable_targets, dtype=np.int32))
            sentences.append(np.array(sentence, dtype=np.int32))
        name_targets = np.array(name_targets, dtype=np.object)
        sentences = np.array(sentences, dtype=np.object)
        code = np.array(code, dtype=np.object)
        target_is_unknown = np.array(target_is_unknown, dtype=np.object)
        copy_vectors = np.array(copy_vectors, dtype=np.object)
        return name_targets, sentences, code, target_is_unknown, copy_vectors

    def validated_rec_copy_conv_data(inp, percent_train, min_code_size):
        assert percent_train < 1
        assert percent_train > 0
        names, code, original_names = FormatTokens.__read_file(inp)
        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(percent_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = FormatTokens(names[idxs[:lim]], code[idxs[:lim]])
        return naming.rec_copy_conv_data(names[idxs[:lim]], code[idxs[:lim]], min_code_size),\
                naming.rec_copy_conv_data(names[idxs[lim:]], code[idxs[lim:]], min_code_size), naming

    def possible_suggestions_from_name_prefix(self, next_name_log_probability, name_cx_size, max_predicted_id_size=5, max_steps=100):
        """
        A list of tuple of full suggestions (token, prob)
        """
        suggestions = defaultdict(lambda: float('-inf'))
        """
        A stack of partial suggestion in the form ([subword1, subword2, ...], logprob)
        - list of subwords and their individual probabilities
        """
        possible_suggestions_stack = [([self.NONE] * (name_cx_size - 1) + [self.SUBTOKEN_START], [], 0)]
        """
        Keep the max_no_of_suggestions suggestion scores (sorted in the heap). 
        Prune further exploration if something has already lower score
        """
        prediction_probabilities_heap = [float('-inf')]
        max_no_of_suggestions = 20
        nsteps = 0
        while True:
            scored_list = []
            while len(possible_suggestions_stack) > 0:
                subword_tokens = possible_suggestions_stack.pop()
                """
                If we're done, append to full suggestions
                """
                if subword_tokens[0][-1] == self.SUBTOKEN_END:
                    final_prediction = tuple(subword_tokens[1][:-1])
                    if len(final_prediction) == 0:
                        continue
                    suggestion_log_probability = np.logaddexp(suggestions[final_prediction], subword_tokens[2])
                    if suggestion_log_probability > prediction_probabilities_heap[0] and not suggestion_log_probability == float('-inf'):
                        """
                        Push only if the score is better than the current minimum and > 0
                        Remove extraneous entries
                        """
                        suggestions[final_prediction] = suggestion_log_probability
                        heapq.heappush(prediction_probabilities_heap, suggestion_log_probability)
                        if len(prediction_probabilities_heap) > max_no_of_suggestions:
                            heapq.heappop(prediction_probabilities_heap)
                    continue
                elif len(subword_tokens[1]) > max_predicted_id_size:
                    continue
    
                """
                Convert subword context
                """
                context = [self.name_dictionary.is_id_or_is_unknown(k) for k in subword_tokens[0][-name_cx_size:]]
                assert len(context) == name_cx_size
                context = np.array([context], dtype=np.int32)
    
                """
                Predict next subwords
                """
                target_subword_log_probabilities = next_name_log_probability(context)
    
                def list_possible_options(name_id):
                    subword_name = self.all_tokens_dictionary.token_from_id(name_id)
                    if subword_name == self.all_tokens_dictionary.get_unknown():
                        subword_name = "***"
                    name = subword_tokens[1] + [subword_name]
                    return subword_tokens[0][1:] + [subword_name], name, target_subword_log_probabilities[0, name_id] + \
                           subword_tokens[2]

                top_indices = np.argsort(-target_subword_log_probabilities[0])
                possible_options = [list_possible_options(top_indices[i]) for i in xrange(max_no_of_suggestions)]
                """
                Remove suggestions that contain duplicate subtokens
                """
                scored_list.extend(filter(lambda x: len(x[1])==1 or x[1][-1] != x[1][-2], possible_options))
            """
            Prune
            """
            scored_list = filter(lambda suggestion: suggestion[2] >= prediction_probabilities_heap[0] and suggestion[2] >= float('-inf'), scored_list)
            scored_list.sort(key=lambda entry: entry[2], reverse=True)
            """
            Update
            """
            possible_suggestions_stack = scored_list[:max_no_of_suggestions]
            nsteps += 1
            if nsteps >= max_steps:
                break
        """
        Sort and append to final predictions
        """
        suggestions = [(identifier, np.exp(logprob)) for identifier, logprob in suggestions.items()]
        suggestions.sort(key=lambda entry: entry[1], reverse=True)
        return suggestions

