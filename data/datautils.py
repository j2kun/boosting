import os

def datasetFilenames(datasetName):
   prefix = os.path.join(os.path.dirname(__file__), datasetName)
   return prefix + '.train', prefix + '.test'

def vectorize(value, values):
   return [int(v==value) for v in values]
