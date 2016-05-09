from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
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
        for tup in expected:
            if round(tup[1])*2 == 0:
                t = 1
            else:
                t = round(tup[1])*2
            trueValues.append(t)
            if round(avg_expected[str(tup[0])]) == 0:
                t = 1
            else:
                t = round(avg_expected[str(tup[0])])
            predictedValues.append(t)
        trueValues = np.array(trueValues)
        predictedValues = np.array(predictedValues)
        for i in range(len(trueValues)):
            if trueValues[i] == 0:
                trueValues[i] = 1
            if predictedValues[i] == 0:
                predictedValues[i] = 1
        print precision_recall_fscore_support(trueValues, predictedValues,average='macro')
        print self.calculateRecall(trueValues, predictedValues)
        print self.calculatePrecision(trueValues, predictedValues)
        
    def calculateRecall(self, trueValues, predictedValues):
        return recall_score(trueValues, predictedValues, average = 'macro')

    def calculatePrecision(self, trueValues, predictedValues):
        return precision_score(trueValues, predictedValues, average = 'macro')

    def calculateF1(self, trueValues, predictedValues):
        return f1_score(trueValues, predictedValues, average = 'macro')

    def calculateAccuracy(self, trueValues, predictedValues):
        return accuracy_score(trueValues, predictedValues)
