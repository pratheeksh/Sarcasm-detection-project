import knnclassifier as kc
import pandas as pd
import patternExtract as pe
import patternMatcher as patternMatcher
import modifiedknn as mKNN
import re
import string
import evaluationMetrics
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
    test_twitter_data,expected = pm.match_test_patterns(cwset,hfwset)
    train2,target2,test2 = knnobj.extract_features_train(scores_features,test_twitter_data)
    output2 = knnobj.classify(train2,target2,test2)
    ev = evaluationMetrics.Evaluation()
    twitter_out = []
    for i,row in enumerate(output2):
        twitter_out.append([row[1][0],test_twitter_data[i]])
    #Modified KNN
    ev.evaluate(twitter_out,expected)
    mknnObj=mKNN.ModifiedKNN()
    predictions=mknnObj.predict_test(train,target,test)
    for i in range(len(output)):
        print "Finally found a Sarcastic Sentence={}  KNNClassifier score={} Modified K Score={}".format(test_data[i]['Text'], output[i][1],predictions[i])

def main():
    sc=SarcasmClassifier()

if __name__ == '__main__':
    main()
