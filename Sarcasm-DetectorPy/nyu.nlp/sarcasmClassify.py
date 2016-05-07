import knnclassifier
import pandas as pd
import patternExtract as pe
import patternMatcher as patternMatcher
import re
import string

class SarcasmClassifier():
    #Load files
    reviews=pd.read_csv("../data/test.csv")
    amazon=pd.read_csv("../data/amazon.csv")

    # Pattern Extraction -
    pattext = pe.PatternExtraction(reviews['text'])
    dic = pattext.calculateCorpusFrequency()
    cwset = pattext.findCW()
    hfwset=pattext.findHFW()

    # Pattern Matching -
    pm =patternMatcher.load_csv()

    #scores_features = [Text,ReviewId,Score]
    scores_features=pm.init(cwset,hfwset)
    knnclassifier.extract_features_train(scores_features)

    # Featurizer



def main():
    sc=SarcasmClassifier()

if __name__ == '__main__':
    main()
