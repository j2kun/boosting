import random
import math


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


# normalize a distribution
def normalize(weights):
    norm = sum(weights)
    return tuple(m / norm for m in weights)


def sign(x):
    return 1 if x >= 0 else -1

