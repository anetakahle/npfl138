#!/usr/bin/env python3
import argparse
import collections
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    data_distribution = collections.Counter()
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_distribution[line] += 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    total = sum(data_distribution.values())
    for key in data_distribution:
        data_distribution[key] /= total

    data_dist = np.array(list(data_distribution.values()))

    # TODO: Load model distribution, each line `string \t probability`.
    model_distribution = dict()
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            string, probability = line.split("\t")
            model_distribution[string] = float(probability)
    # TODO: Create a NumPy array containing the model distribution.
    model_dist = np.array([model_distribution.get(k, 0) for k in data_distribution.keys()])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_dist * np.log(data_dist))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    if 0 in model_dist:
        return entropy, np.inf, np.inf

    crossentropy = -np.sum(data_dist * np.log(model_dist))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy


    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
