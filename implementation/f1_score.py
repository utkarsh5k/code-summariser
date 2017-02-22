from collections import defaultdict
import heapq
import os
from scipy.integrate import simps
import os.path
import numpy as np

class F1Evaluator:
    def __init__(self, model):
        self.model = model
        self.max_predicted_identifier_size = 6

    def compute_names(self, features, targets, token_dict):
        #Top suggestions for a paragraph vector

        result_accumulator = PointSuggestionEvaluator()
        for i in xrange(features.shape[0]):
            #extract result from 2d view of features[]
            result = self.model.predict_name(np.atleast_2d(features[i]))
            confidences = [suggestion[1] for suggestion in result]
            #match with real targets, store true or false prediction
            is_correct = [','.join(suggestion[0]) == targets[i] for suggestion in result]
            #out of vocabulary suggestion
            is_unkd = [is_unk(''.join(suggestion[0])) for suggestion in result]
            unk_word_accuracy = [self.unk_acc(suggestion[0], targets[i].split(','), token_dictionary) for suggestion in result]
            # precision and recall for the suggestions
            precision_recall = [token_precision_recall(suggestion[0], targets[i].split(',')) for suggestion in result]
            #combine metrics to form a single result
            result_accumulator.add_result(confidences, is_correct, is_unkd, precision_recall, unk_word_accuracy)

        return result_accumulator

    def unk_acc(self, suggested_subtokens, real_subtokens, token_dictionary):
        #real target values which are OOV
        real_unk_subtokens = set(t for t in real_subtokens if t not in token_dictionary)
        if len(real_unk_subtokens) == 0:
            return None
        #suggested values present in OOV target values
        return float(len([t for t in suggested_subtokens if t in real_unk_subtokens])) / len(real_unk_subtokens)

def is_unk(joined_tokens):
    return ["*"] * len(joined_tokens) == joined_tokens

#class PointSuggestionEvaluator:


def token_precision_recall(predicted_parts, target_parts):
    """
    Get the precision/recall for the given token.

    :param predicted_parts: a list of predicted parts
    :param target_parts: a list of the golden parts
    :return: precision, recall, f1 as floats
    -- precision = tp / (tp + fp)
    -- recall = tp / (tp + fn)
    -- f1 = 2.((precision.recall) / (precison + recall))
    -- tp: true positives, fp: false positives, fn: false negatives
    """
    ground = [tok.lower() for tok in target_parts]

    tp = 0
    for subtoken in set(predicted_parts):
        if subtoken == "***" or subtoken is None:
            continue  # Ignore UNKs
        if subtoken.lower() in ground:
            #remove since already predicted, dont need extra tps
            ground.remove(subtoken.lower())
            tp += 1

    assert tp <= len(predicted_parts), (tp, len(predicted_parts))
    if len(predicted_parts) > 0:
        """ Predicted parts is basically true and false positives """
       precision = float(tp) / len(predicted_parts)
    else:
       precision = 0

    assert tp <= len(gold_set_parts), (tp, gold_set_parts)
    if len(target_parts) > 0:
        """ Target tokens contain tp and misclassified fn """
       recall = float(tp) / len(target_parts)
    else:
       recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.
    return precision, recall, f1
