from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
    def evaluate_results(self,expected,predicted):

        if(len(expected)!=len(predicted)):
            print "Looks like some values were not predicted properly"
        print "Precision, Accuracy, Recall, FScore"
        print precision_recall_fscore_support(expected, predicted,average='binary')

        cm=confusion_matrix(expected, predicted)
        print "Confusion matrix"
        print cm


    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks =[0,1]
        plt.xticks(tick_marks, ["Sarcastic","Non-Sarcastic"], rotation=45)
        plt.yticks(tick_marks, ["Sarcastic","Non-Sarcastic"])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    def calculateRecall(self, trueValues, predictedValues):
        return recall_score(trueValues, predictedValues, average = 'macro')

    def calculatePrecision(self, trueValues, predictedValues):
        return precision_score(trueValues, predictedValues, average = 'macro')

    def calculateF1(self, trueValues, predictedValues):
        return f1_score(trueValues, predictedValues, average = 'macro')

    def calculateAccuracy(self, trueValues, predictedValues):
        return accuracy_score(trueValues, predictedValues)
