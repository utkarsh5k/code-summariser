# python unknown_token_stats.py train test

import json
import sys
from collections import Counter
from itertools import chain

def tokens(inp):
    with open(inp) as f:
        data = json.load(f)
    return Counter(chain.from_iterable((datum["name"] for datum in data)))

if __name__ == "__main__":

    train_tokens = tokens(sys.argv[1])
    test_tokens = tokens(sys.argv[2])

    known_tokens = set(t for t, c in train_tokens.iteritems() if c>1)
    no_of_unknown_tokens = sum(c for t, c in test_tokens.iteritems() if t not in known_tokens)
    total_test_tokens = sum(test_tokens.values())

    print "UNK toks in test %s = (%s/%s)" % (float(no_of_unknown_tokens)/total_test_toks , no_of_unknown_tokens, total_test_tokens)

