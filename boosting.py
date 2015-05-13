import math
from utils import draw, normalize, sign


# compute the weighted error of a given hypothesis on a distribution
def computeError(h, examples, weights):
   hypothesisResults = [h(x)*y for (x,y) in examples] # +1 if correct, else -1
   return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)


# boost: [(list, label)], learner, int -> (list -> label)
# where a learner is (() -> (list, label)) -> (list -> label)
# boost the weak learner into a strong learner
def boost(examples, weakLearner, rounds):
   distr = normalize([1] * len(examples))
   hypotheses = [None] * rounds
   alpha = [0] * rounds

   for t in range(rounds):
      def drawExample():
         return examples[draw(distr)]

      hypotheses[t] = weakLearner(drawExample)
      hypothesisResults, error = computeError(hypotheses[t], examples, distr)

      alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
      distr = normalize([d * math.exp(-alpha[t] * h)
                         for (d,h) in zip(distr, hypothesisResults)])
      print("Round %d, error %.3f" % (t, error))

   def finalHypothesis(x):
      return sign(sum(a * h(x) for (a, h) in zip(alpha, hypotheses)))

   return finalHypothesis, distr
