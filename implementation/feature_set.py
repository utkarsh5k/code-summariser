from collections import Counter

"""
Feature representation, converting tokens to ids and ids to tokens
"""
class FeatureSet:

    def __init__(self):
        self.id_to_token = {}
        self.next_id = 0
        self.token_to_id = {}
        self.id_from_token(self.get_unknown())

    def token_from_id(self, id):
        return self.id_to_token[id]

    def is_id_or_is_unknown(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        return self.token_to_id[self.get_unknown()]

    def id_from_token(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        present_id = self.next_id
        self.next_id = self.next_id + 1
        self.token_to_id[token] = present_id
        self.id_to_token[present_id] = token
        return present_id

    def is_unknown(self, token):
        return token not in self.token_to_id

    @staticmethod
    def get_unknown():
        return "%UNK%"

    def __len__(self):
        return len(self.token_to_id)

    def __str__(self):
        return str(self.token_to_id)

    def is_id_or_is_none(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        return None

    def get_all_names(self):
        """
        use of frozenset function since it is immutable and then only can it serve as key in a dictionary
        """
        return frozenset(self.token_to_id.keys())

    @staticmethod
    def make_feature_set(tokens, min_occurences=20):
        """
        Counter subclass needed to count hashable objects (json/dictionary being built for tokens)
        """
        occurences = Counter(tokens)
        feature_set = FeatureSet()
        for token, count in occurences.iteritems():
            if count >= min_occurences:
                feature_set.id_from_token(token)
        return feature_set
