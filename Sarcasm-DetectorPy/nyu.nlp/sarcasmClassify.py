
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
    cwset.__contains__()
    # Pattern Matching -
    pm =patternMatcher.load_csv()
    testpatterns=pm.generate_all_patterns(reviews['text'],cwset,hfwset)

    # Featurizer
    print testpatterns.__str__()


def main():
    sc=SarcasmClassifier()

if __name__ == '__main__':
    main()