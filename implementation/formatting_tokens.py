from feature_set import feature_set
import json
from itertools import chain
from collections import defaultdict

class FormatTokens:

	SUBTOKEN_START = "%START%"
    SUBTOKEN_END = "%END%"
    NONE = "%NONE%"

    def __init__(self, names, code):
        self.name_dictionary = FeatureSet.make_feature_set(chain.from_iterable(names), 2)
        self.name_dictionary.id_from_token(self.NONE)

        self.all_tokens_dictionary = FeatureSet.make_feature_set(chain.from_iterable([chain.from_iterable(code), chain.from_iterable(names)]), 5)
        self.all_tokens_dictionary.id_from_token(self.NONE)

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
