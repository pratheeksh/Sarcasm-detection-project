import csv
import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
import extractsarcastic

reload(sys)
sys.setdefaultencoding('utf8')

class knnClassifier():
### Define specific features
    def extract_features_train(self,traindata,testdata):
        train,target,test = [],[],[]

        for data in testdata:
            test.append(self.extract_features_sentence(data))
        for data in traindata:
            train.append(self.extract_features_sentence(data))
            target.append(round(float(data['Score'])*100))
        return train,target,test

    def extract_features_sentence(self,data):

        feature_vector=[ data['Text'].count('!'),
                 data['Text'].count('?'),
                 len(data['Text']),
                 float(data['Score']),
                 float(data['Funny Score']),
                data['Text'].count('\"')]+extractsarcastic.identify_sentiment(str(data['Text']))
        print feature_vector
        return feature_vector

    def classify(self,features,target,test):
        #print features,target
        output = []
        neigh = KNeighborsClassifier(n_neighbors=10)
        neigh.fit(np.array(features), np.array(target))
        for vector in test:
            output.append([vector,neigh.predict([vector])])
        return output



