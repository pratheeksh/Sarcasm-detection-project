import re
import string
import nltk
from nltk.tag import pos_tag
import pandas as pd
import pickle

class PatternExtraction:
	pnSet=[]
	wordFreqDict=[]
	def __init__(self, reviews):
		self.reviewList = reviews
	def loadPropernouns(self,reviews):
		for each_review in reviews:
			tagged_sent = pos_tag(each_review.split())
			propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
			for each_pn in propernouns:
				if each_pn in self.pnSet:
					continue
				else:
					self.pnSet.append(each_pn)

	def calculateCorpusFrequency(self):
	#pass 1: for each word in each review, find total number of occurrences.
	#also find total number of words in the overall corpus of reviews.
		wordFreqDict = {}
		totalNumWordsInCorpus = 0
		self.loadPropernouns(self.reviewList)

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
		pickle.dump(wordFreqDict, open( "wordfreqdict.p", "wb" ) )
		return wordFreqDict

	def findCW(self,wordFreqDict):
		cwSet = set()
		#upperbound for fc = 1000 words per million
		fcThresholdMax = (float(1000) / 1000000)
		#wordFreqDict=pickle.load( open( "wordfreqdict.p", "rb" ) )
		for word in wordFreqDict:
			#punctuation is not CW
			if word in string.punctuation:
				continue
			if word in cwSet:
				continue
			corpusFreqWord = wordFreqDict[word]
			if (corpusFreqWord < fcThresholdMax and word not in self.pnSet):
				cwSet.add(word)
		return cwSet

	def findHFW(self,wordFreqDict):
		hfwSet = set()
		#lowerbound for hfw = 1000 words per million
		fwThresholdMin = (float(1000) / 1000000)
		#wordFreqDict=pickle.load( open( "wordfreqdict.p", "rb" ) )
		for word in wordFreqDict:
			if word in hfwSet:
				continue
			corpusFreqWord = wordFreqDict[word]
			if (corpusFreqWord > fwThresholdMin or word in self.pnSet):
				hfwSet.add(word)
		return hfwSet



def main():
	reviews=pd.read_csv("../data/amazon.csv")
	pattext = PatternExtraction(reviews['Text'])
	dic = pattext.calculateCorpusFrequency()
	#Pickle the proper noun set

	pattext.findCW()

if __name__ == "__main__":
	main()