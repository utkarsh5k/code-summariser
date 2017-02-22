# python token_statistics.py input

import json
import sys
from collections import Counter
from itertools import chain

if __name__ == "__main__":

    inp = sys.argv[1]
    with open(inp, 'r') as f:
        data = json.load(f)

    no_of_names = Counter(chain.from_iterable((datum["name"] for datum in data)))

    max_no_of_rare = 10

    in_name_rare = {True: 0, False:0}
    also_present_in_body_rare = {True: 0, False: 0}

    for datum in data:
        token_names = set(datum["name"])
        subtokens = set(datum["tokens"])

        for t in token_names:
            is_rare = no_of_names[t] < max_no_of_rare
            in_name_rare[is_rare]+=1
            if t in subtokens:
                also_present_in_body_rare[is_rare]+=1

    """
    Use of true positives and negatives for rare and common occurences of tokens
    """
    print "Rare: %s" % (float(also_present_in_body_rare[True]) / in_name_rare[True])
    print "Common: %s" % (float(also_present_in_body_rare[False]) / in_name_rare[False])