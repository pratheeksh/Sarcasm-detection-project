from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
class Evaluation:
    def evaluate(self,output,expected):
        d={}
        c={}
        avg_expected = {}
        trueValues = []
        predictedValues = []
        for row in output:
            score = float(row[0])
            review_id = row[1]['Review_id']
            if review_id not in d:
                d[review_id] = score
                c[review_id] = 1
            else:
                d[review_id]+= score
                c[review_id]+= 1
        for rid in d:
            avg_score = d[rid]/c[rid]
            avg_expected[rid] = avg_score
        print avg_expected
        print expected
        for tup in expected:
            trueValues.append(tup[1])
            predictedValues.append(avg_expected[str(tup[0])])
        print self.calculateRecall(np.array(trueValues), np.array(predictedValues))


    def calculateRecall(self, trueValues, predictedValues):
        return recall_score(trueValues, predictedValues, average = 'macro')

    def calculatePrecision(self, trueValues, predictedValues):
        return precision_score(trueValues, predictedValues, average = 'macro')

    def calculateF1(self, trueValues, predictedValues):
        return f1_score(trueValues, predictedValues, average = 'macro')

    def calculateAccuracy(self, trueValues, predictedValues):
        return accuracy_score(trueValues, predictedValues)
