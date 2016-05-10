import re
from sklearn.feature_extraction.text import CountVectorizer

def generateUnigrams(reviewsCollectionTrainSet):
	vc = CountVectorizer(stop_words = 'english')
	vc.fit_transform(reviewsCollectionTrainSet).toarray()
	return vc

def generateUnigramFeatureVect(review, vec):
	return vec.transform(review).toarray().tolist()[0]

def listify(review):
	wrapperList = []
	wrapperList.append(review)
	return wrapperList

def main():
	coll = ['this is a review', 'dollface profile picture', 'definitely more coffee please']
	vc = generateUnigrams(coll)
	#print vc.get_feature_names()
	sentence = 'review profile'
	trainFeatureVector = generateUnigramFeatureVect(listify(sentence), vc)
	print trainFeatureVector

if __name__ == "__main__":
	main()
