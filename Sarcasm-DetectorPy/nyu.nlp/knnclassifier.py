import csv
import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
import extractsarcastic
import feature_extracter as fe
import sentiments as senti
import load_sent
sentiments  = load_sent.load_sent_word_net()
reload(sys)
sys.setdefaultencoding('utf8')

class knnClassifier():
### Define specific features
    ldamodel=None
    def extract_features_train(self,traindata,testdata,topic, feat):

        train,target,test = [],[],[]
        for index,data in testdata.iterrows():
            test.append(self.extract_features_sentence(data,topic,feat))
        for index,data in traindata.iterrows():
            train.append(self.extract_features_sentence(data,topic,feat))
            target.append((data['Score']))
        return train,target,test

    def extract_features_sentence(self,df,topic,feat):
        fv=fe.FeatureExtractor()
        data=df['Text']
        sent = senti.Sentiment()

        feature_vector=[ fv.count_apost(data),
                         fv.count_qn(data),
                         fv.data_len(data),
                         fv.count_quotes(data),
                         #fv.sarcastic_score(data),
                         fv.count_capitals(data)] + fv.get_topic(data,topic) + feat.generateUnigramFeatureVe(data) +  sent.calculateSent(data,sentiments)
        return feature_vector


    def classify(self,features,target,test):
        #print features,target
        output = []
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(np.array(features), np.array(target))
        for vector in test:
            output.extend(neigh.predict([vector]))
        return output



