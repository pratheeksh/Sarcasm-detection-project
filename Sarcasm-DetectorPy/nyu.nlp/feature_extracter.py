import unigrams as unigrams
import nltk
import extractsarcastic
import numpy as np
import topics as topic
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction import DictVectorizer
class FeatureExtractor():
    ldamodel=None
    vec = None
    def bigram_word_feats(self,words):
        score_fn = BigramAssocMeasures.chi_sq
        n = 200
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        vec = DictVectorizer()
        measurements = dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
        vec.fit_transform(measurements).toarray()
    def generateUnigramVectorizer(self, dataCollection):
        self.vec = unigrams.generateUnigrams(dataCollection)

    def generateUnigramFeatureVe(self, review):
	#vec is obtained from above func on all reviews
	#this returns feature vec for one review
	listReview = unigrams.listify(review)
	return unigrams.generateUnigramFeatureVect(listReview, self.vec)

    def generateDictionary(data):
        '''
        This function identifies unique n-grams in your data.
        '''
        vocab = set()
        for line in data:
            for item in line:
                vocab.add(item)
        dictionary = {}
        i=0
        for item in vocab:
            dictionary[item] = i
            i+=1
        return dictionary
    def doc2Bow(bigramData, dictionary):
        vect = [0]*len(dictionary) # Initialize vector to zero
        for gram in bigramData:
            vect[dictionary[gram]]+=1
        return np.asarray(vect)  # Convert to numpy vector

    def getStringBigrams(string):
        if len(string) <= 0: return []
        if len(string) == 1: return string[0] # Handle strings with only one character = extract unigram
        return [string[i]+string[i+1] for i in range(len(string)-1)]

    def getDataBigrams(strings):
        return [FeatureExtractor.getStringBigrams(x) for x in strings]
    def count_apost(self,text):
        return text.count('!')
    def count_qn(self,text):
        return text.count('?')
    def count_capitals(self,text):
        count=0
        tokens=nltk.word_tokenize(text)
        for each_word in tokens:
            if each_word[0].isupper():
                count+=1
        if count>4:
            return 1
        return 0
    def get_topic(self,text,topic):
        topic_vector=list(topic.getReviewTopicVector(text))
        tvector=[]

        for i in range(20):
            tvector.append(0)
        for num,each_topic in topic_vector:
            tvector[num]=each_topic

        return tvector
    def data_len(self,text):
        return len(text)
    def count_quotes(self,text):
        return text.count('\"')
    def sarcastic_score(self,text):

        return extractsarcastic.identify_sentiment(text)

    def extract_all_features(self,trainDataFrame,testDataFrame):
        train=[]
        test=[]

        train = ([(self.extract_feature_dict(row), row['SASI']) for index, row in trainDataFrame.iterrows()])
        print "++++++ Extracted TRAIN features"

        test= ([(row['Text']) for index, row in testDataFrame.iterrows()])
        print "++++++ Extracted TEST features"
        return train,test

