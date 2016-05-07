import knnclassifier as kc
import pandas as pd
import patternExtract as pe
import patternMatcher as patternMatcher
import modifiedknn as mKNN
import re
import string

class SarcasmClassifier():
    #Load files
    reviews=pd.read_csv("../data/test.csv")
    amazon=pd.read_csv("../data/amazon.csv")

    # Pattern Extraction -
    print "+++++++++ Extracting patterns +++++++++"
    pattext = pe.PatternExtraction(reviews['text'])
    dic = pattext.calculateCorpusFrequency()
    cwset = pattext.findCW()
    hfwset=pattext.findHFW()
    print "+++++++++ Pattern matching +++++++++"
    # Pattern Matching -
    pm =patternMatcher.load_csv()

    #scores_features = [Text,ReviewId,Score]
    scores_features,test_data=pm.init(cwset,hfwset)
    i=pm.init(cwset,hfwset)


    print "++++++++ KNN Classification starts here  +++++++"
    knnobj=kc.knnClassifier()
    train,target,test = knnobj.extract_features_train(scores_features,test_data)
    output = knnobj.classify(train,target,test)


    #Modified KNN
    mknnObj=mKNN.ModifiedKNN()
    predictions=mknnObj.predict_test(train,target,test)
    for i in range(len(output)):
        print "Sentence={}  KNNClassifier score={} Modified K Score={}".\
            format(test_data[i]['Text'], output[i][1],predictions[i])

def main():
    sc=SarcasmClassifier()

if __name__ == '__main__':
    main()
