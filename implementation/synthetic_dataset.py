"""
create synthetic dataset using tokens and mapping as seen below
"""

import json
import random

unordered_names = {
    frozensetter(["P"]): ["hello"],
    frozensetter(["P", "Q"]): ["hello", "goodbye"],
    frozensetter(["P", "Q", "R"]): ["hello", "barack"],
    frozensetter(["P", "R"]): ["hello", "racism"],
    frozensetter(["Q"]): ["setter", "goodbye"],
    frozensetter(["Q", "R"]): ["initialize"],
    frozensetter(["R"]): ["chill", "nice"],
    frozensetter(["R", "S"]): ["chill", "test"],
    frozensetter(["D"]): ["had", "test"],
    frozensetter(["P", "S"]): ["hello", "test"],
    frozensetter(["Q", "S"]): ["yeah", "racism"]
}

ordered_names = {
    ("P", "Q"): ["hello"],
    ("Q", "P"): ["setter"],
    ("P", "Q", "R"): ["hello", "racism"],
    ("Q", "P", "R"): ["setter", "racism"],
    ("Q", "R"): ["chill", "racism"],
    ("R", "Q"): ["yeah", "nice"],
    ("R", "P"): ["yeah", "racism"],
    ("P", "R"): ["random"],
    ("R", "P", "Q"): ["yeah", "test"],
    ("P", "D", "Q"): ["hello", "cold"], # S is position invariant
    ("P", "Q", "S"): ["hello", "cold"],
    ("S", "P", "Q"): ["hello", "cold"],
    ("Q", "S", "P"): ["setter", "cold"],
    ("Q", "P", "S"): ["setter", "cold"],
    ("S", "Q", "P"): ["setter", "cold"],
    ("S"): ["cold"],
    ("Q", "R", "S"): ["chill", "cold"],
    ("S", "Q", "R"): ["chill", "cold"],
    ("Q", "S", "R"): ["chill", "cold"],
    ("R", "Q", "S"): ["yeah", "cold"],
    ("S", "R", "Q"): ["yeah", "cold"],
    ("R", "S", "Q"): ["yeah", "cold"],
    ("S", "M"): ["code"],
    ("M", "S"): ["random"],
    ("M", "P", "Q"): ["hello", "random", "code"],
    ("P", "M", "Q"): ["hello", "random", "code"],
    ("M", "Q", "P"): ["setter", "random", "code"],
    ("Q", "P", "M"): ["setter", "random", "code"]
}

def unordered_dataset_generator(no_of_samples, noise=0.8):
	records = []
    for i in xrange(no_of_samples):
        current_records = unordered_names.keys()[random.randint(0, len(unordered_names) - 1)]
        name = unordered_names[current_records]
        tokens = []
        already_present_records = set()
        additional_noise = random.random() < noise
        while additional_noise or len(already_present_records) != len(current_records):
            if additional_noise:
                record = str(random.randint(0, 100))
            else:
                record = [t for t in current_records][random.randint(0, len(current_records) - 1)]
                already_present_records.update(record)
            tokens.append(record)
            additional_noise = random.random() < noise
        records.append({"tokens": tokens, "name": name})
    return records

def ordered_dataset_generator(no_of_samples, noise = 0.7):
	records = []
    for i in xrange(no_of_samples):
        current_records = ordered_names.keys()[random.randint(0, len(ordered_names) - 1)]
        name = ordered_names[current_records]
        tokens = []
        current_idx = 0
        additional_noise = random.random() < noise
        while additional_noise or current_idx < len(current_records):
            if additional_noise:
                record = str(random.randint(0, 100))
            else:
                record = current_records[current_idx]
                if random.random() < noise:
                    current_idx += 1
            tokens.append(record)
            additional_noise = random.random() < noise
        records.append({"tokens": tokens, "name": name})
    return records

"""
generating train test data at 80%
"""
if __name__ == "__main__":
    with open('synthetic_train.json', 'w+') as f:
        json.dump(ordered_dataset_generator(8000), f)
    with open('synthetic_test.json', 'w+') as f:
        json.dump(ordered_dataset_generator(2000), f)
