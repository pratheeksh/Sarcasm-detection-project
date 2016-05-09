
import pandas as pd
import feature_extract_yelp as fey
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from textblob.classifiers import DecisionTreeClassifier
from textblob import TextBlob
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import Extractors.CapitalizedWordLengthExtractor as cwl
from sklearn.pipeline import FeatureUnion
class SVMClassify():

    def init(self):
        print "Classification through SVM"
        train,target,test=self.load_data()
        #train_feature,test_feature=fey.FeatureExtractor().extract_all_features(train,test)
        return train,target,test
    def load_data(self):
        train=pd.read_csv("../data/amazon.csv")
        target= train['SASI'].apply(lambda x: "SARC" if x >= 3 else "NOT")
        # reddit=pd.read_csv("../data/reddit.csv")
        test=pd.read_csv("../data/test.csv")
        return train,target,test

    def setup_classifier(self,train,target,test):


        pipeline = Pipeline([
            ('feats', FeatureUnion([
                ('vectorizer',  CountVectorizer()), # can pass in either a pipeline
                ('ave', cwl.CapitalizedWordLengthExtractor()) # or a transformer
                ]))
         ,('svc', SVC(kernel='linear'))])
        categories = ['SARC', 'NOT']
        print train['Text']
        pipeline.fit(train['Text'],target)

        predicted=pipeline.transform(test['Text'])
        return predicted


def main():
    svm=SVMClassify()
    train,target,test=svm.init()

    print "Extracted features sets for train and set. "
    print "Setting up the classifier"

    predicted=svm.setup_classifier(train,target,test)
    print predicted

if __name__ == '__main__':
    main()