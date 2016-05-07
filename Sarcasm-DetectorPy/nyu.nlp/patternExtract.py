import re
import string

class PatternExtraction:
	def __init__(self, reviews):
		self.reviewList = reviews

	def calculateCorpusFrequency(self):
	#pass 1: for each word in each review, find total number of occurrences.
	#also find total number of words in the overall corpus of reviews.
		wordFreqDict = {}
		totalNumWordsInCorpus = 0

		for review in self.reviewList:
			#treats punctuation and word as a word
			wordList = re.findall(r"[\w']+|[.,!?;]", review)
			for word in wordList:
				totalNumWordsInCorpus += 1
				if word not in wordFreqDict:
					wordFreqDict[word] = 1
				else:
					wordCount = wordFreqDict[word]
					wordFreqDict[word] = wordCount + 1

		#pass 2: calculate the corpus-freq of each word
		for review in self.reviewList:
			wordList = re.findall(r"[\w']+|[.,!?;]", review)
			for word in wordList:
				wordCount = wordFreqDict[word]
				#word already normalized; skip over it
				if wordCount < 1:
					continue
				normalizedCount = (float(wordCount) / totalNumWordsInCorpus)
				wordFreqDict[word] = normalizedCount
		#wordFreqDict now stores normalized word freq
		return wordFreqDict

	def findCW(self):
		cwSet = set()
		wordFreqDict = self.calculateCorpusFrequency()
		#upperbound for fc = 1000 words per million
		fcThresholdMax = (float(1000) / 1000000)
		for word in wordFreqDict:
			#punctuation is not CW
			if word in string.punctuation:
				continue
			if word in cwSet:
				continue
			corpusFreqWord = wordFreqDict[word]
			if (corpusFreqWord < fcThresholdMax):
				cwSet.add(word)
		return cwSet

	def findHFW(self):
		hfwSet = set()
		wordFreqDict = self.calculateCorpusFrequency()
		#lowerbound for hfw = 1000 words per million
		fwThresholdMin = (float(1000) / 1000000)
		for word in wordFreqDict:
			if word in hfwSet:
				continue
			corpusFreqWord = wordFreqDict[word]
			if (corpusFreqWord > fwThresholdMin):
				hfwSet.add(word)
		return hfwSet


def main():
	lis = ['this is a review','this is another review','i do not like avacados','casper mattresses are expensive!']
	pattext = PatternExtraction(lis)
	dic = pattext.calculateCorpusFrequency()
	cwset = pattext.findCW()

	print (cwset)

if __name__ == "__main__":
	main()