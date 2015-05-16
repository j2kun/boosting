import sys

class Stump:
   def __init__(self):
      self.gtLabel = None
      self.ltLabel = None
      self.splitThreshold = None
      self.splitFeature = None

   def classify(self, point):
      if point[self.splitFeature] >= self.splitThreshold:
         return self.gtLabel
      else:
         return self.ltLabel

   def __call__(self, point):
      return self.classify(point)


def majorityVote(data):
   ''' Compute the majority of the class labels in the given data set. '''
   labels = [label for (pt, label) in data]
   try:
      return max(set(labels), key=labels.count)
   except:
      return -1


def minLabelErrorOfHypothesisAndNegation(data, h):
   posData, negData = ([(x, y) for (x, y) in data if h(x) == 1],
                       [(x, y) for (x, y) in data if h(x) == -1])

   posError = sum(y == -1 for (x, y) in posData) + sum(y == 1 for (x, y) in negData)
   negError = sum(y == 1 for (x, y) in posData) + sum(y == -1 for (x, y) in negData)
   return min(posError, negError) / len(data)


def bestThreshold(data, index, errorFunction):
   '''Compute best threshold for a given feature. Returns (threshold, error)'''

   thresholds = [point[index] for (point, label) in data]
   def makeThreshold(t):
      return lambda x: 1 if x[index] >= t else -1

   errors = [(threshold, errorFunction(data, makeThreshold(threshold))) for threshold in thresholds]
   return min(errors, key=lambda p: p[1])


def defaultError(data, h):
   return minLabelErrorOfHypothesisAndNegation(data, h)


def buildDecisionStump(drawExample, errorFunction=defaultError, debug=True):
   # find the index of the best feature to split on, and the best threshold for
   # that index

   data = [drawExample() for _ in range(500)]

   bestThresholds = [(i,) + bestThreshold(data, i, errorFunction) for i in range(len(data[0][0]))]
   feature, thresh, _ = min(bestThresholds, key = lambda p: p[2])

   stump = Stump()
   stump.splitFeature = feature
   stump.splitThreshold = thresh
   stump.gtLabel = majorityVote([x for x in data if x[0][feature] >= thresh])
   stump.ltLabel = majorityVote([x for x in data if x[0][feature] < thresh])

   if debug:
      sys.stderr.write('Feature: %d, threshold: %d, %s\n' % (feature, thresh, '+' if stump.gtLabel == 1 else '-'))

   return stump
