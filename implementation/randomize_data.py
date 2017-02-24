# python randomize_tokens.py input output

import json
import random
import sys

def randomize_data_by_tokens(infile, outfile):

    with open(infile) as f:
        data = json.load(f)
    
    for function in data:
        tokens = function["tokens"][1:-1]
        random.shuffle(tokens)
        function["tokens"] = function["tokens"][0:1] + tokens + function["tokens"][-1:]

    with open(outfile, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        randomize_data_by_tokens(sys.argv[1], sys.argv[2])