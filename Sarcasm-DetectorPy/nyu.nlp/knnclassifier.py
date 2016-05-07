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
        train = []
        target = []
        test = []
        count =0
        for data in traindata:
            train.append(self.extract_features_sentence(data['Text']))
            target.append(round(float(data['Score'])*100))
        return train,target,testdata
    def extract_features_sentence(self,text):
        print text
        return [ text.count('!'),
                 text.count('?'),
                 len(text),
                text.count('\"')]+extractsarcastic.identify_sentiment(str(text))
    def classify(self,features,target,test):
        #print features,target
        output = []
        neigh = KNeighborsClassifier(n_neighbors=10)
        neigh.fit(np.array(features), np.array(target))
        for sent in test:
            
            vector = self.extract_features_sentence(sent['Text'])
             
            output.append([sent,neigh.predict([vector])])
        return output
    def generate_dataframe(self,filename):
        with open(filename,"rb") as csvfile:
            reader = csv.DictReader(csvfile)
            train_data = []
            for row in reader:
                res = {}
                text = row['Text']
                b, c, d, e = float(row['MT1']), float(row['MT2']), float(row['MT3']), float(row['SASI'])
                avg_score = statistics.mean([b,c,d,e])
                res['text'] = text
                res['score'] = avg_score
                train_data.append(res)

        return train_data

