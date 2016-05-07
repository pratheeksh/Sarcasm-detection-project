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
 


def main():
	true = [1,3,4,5]
	pred = [1,3,3,4]
	eval = Evaluation()
	print eval.calculateRecall(true, pred)
	print eval.calculatePrecision(true, pred)
	print eval.calculateF1(true, pred)
	print eval.calculateAccuracy(true, pred)

if __name__ == "__main__":
	main()
