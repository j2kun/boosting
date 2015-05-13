import random
import boosting
from utils import sign

def target(x):
   if x[2] > 0.5 or x[3] > 0.5:
      return 1 if random.random() > 0.05 else -1
   return -1


examples = [[random.random() for _ in range(10)] for _ in range(1000)]
labels = [target(x) for x in examples]
trainingData = list(zip(examples, labels))

testData = [[random.random() for _ in range(10)] for _ in range(1000)]
testLabels = [target(x) for x in testData]


def testCoordinate(samples, j):
    values = [sign(x[j] - 0.5) * y for (x,y) in samples]
    return len([z for z in values if z > 0]) / len(values)


def bestCoordinate(samples, n):
    return max(range(n), key=lambda j: testCoordinate(samples, j))


# find the single coordinate and a threshold value that works best
def singleCoordinateLearner(drawExample):
    samples = [drawExample() for _ in range(100)]
    n = len(samples[0][0])

    j = bestCoordinate(samples, n)
    return lambda x: x[j] > 0.5


finalH, finalDistr = boosting.boost(trainingData, singleCoordinateLearner, 100)

finalError = len([x for x in testData if finalH(x) != target(x)]) / len(testData)
print(finalError)
