import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
from collections import defaultdict
import nltk

class Topic:
	def __init__(self):
		self.stop = stopwords.words('english')
		self.porter = nltk.PorterStemmer()
		self.dictionary = None
		self.ldamodel=None

	def getLdaModel(self, reviewCollection):
		#this returns the LDA model computed over all the training reviews in the collection
		#filter out stoplist words
		texts = [[self.porter.stem(word) for word in review.lower().split() if word not in self.stop] for review in reviewCollection]
		wordFreq = defaultdict(int)
		for text in texts:
			for token in text:
				wordFreq[token] += 1
		
		texts = [[token for token in text if wordFreq[token] > 1] for text in texts]
		dictionary = corpora.Dictionary(texts)
		self.dictionary = dictionary
		corpus = [dictionary.doc2bow(text) for text in texts]
		self.ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, alpha='auto')



	def getReviewTopicVector(self, reviewSentence):

		text = [self.porter.stem(word) for word in reviewSentence.lower().split() if word not in self.stop]
		wrapperList = []
		wrapperList.append(text)
		reviewBagOfWords = self.dictionary.doc2bow(text)
		return self.ldamodel[reviewBagOfWords]

	def trainTopics(self,data):
		ldamodel =self.getLdaModel(data)
		return ldamodel
def main():
	topc = Topic()

	revs = ['hi this is a rreview', 'this is another review', 'we write another review']
	query = 'this is a wuery string for review'
	topc.getLdaModel(revs)
	ls = topc.getReviewTopicVector(query)
	print ls

	
if __name__ == "__main__":
	main()
