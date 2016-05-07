from nltk.tag import StanfordNERTagger

class NamedEntityRecognizer:
	def __init__(self, pathToModel):
		self.modelPath = pathToModel
	
	def namedEntityRecognize(self, sentence):
	#perform NER on the sentence - returns a list of tuples of (word, ne-recognized tags)
		st = StanfordNERTagger(self.modelPath)
		print st.tag(sentence.split())
		return st.tag(sentence.split())

def main():
	modelPath = "/Users/purnima/Downloads/test/stanford-ner-2015-12-09/classifiers/english.conll.4class.distsim.crf.ser.gz"
	ner = NamedEntityRecognizer(modelPath)
	sentence = "I love Dave and Busters!!  It is a great place to go when you are wanting to have a great time with a great group of friends.  If you find you and your friends bored on a Friday or Saturday night, come to D&B and win some prizes!!  It is pricey though, so bring some major moolah"
	ner.namedEntityRecognize(sentence);

if __name__ == "__main__":
	main()
