from feature_set import feature_set
import json
from itertools import chain
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

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


