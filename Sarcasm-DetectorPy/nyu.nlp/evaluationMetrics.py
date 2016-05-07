from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Evaluation:
	def calculateRecall(self, trueValues, predictedValues):
		return recall_score(trueValues, predictedValues, average = 'macro')

	def calculatePrecision(self, trueValues, predictedValues):
		return precision_score(trueValues, predictedValues, average = 'macro')

	def calculateF1(self, trueValues, predictedValues):
		return f1_score(trueValues, predictedValues, average = 'macro')

	def calculateAccuracy(self, trueValues, predictedValues):
		return accuracy_score(trueValues, predictedValues)
