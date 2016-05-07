import csv
import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import extractsarcastic
class knnClassifier():
### Define specific features
    def extract_features_train(self,traindata):
        train = []
        target = []
        for data in traindata:
            train.append(self.extract_features_sentence(data['Text']))
            target.append(round(float(data['Score'])*100))
        self.classify(train,target)
    def extract_features_sentence(self,text):
        return [ text.count('!'),
                 text.count('?'),
                 len(text),
                text.count('\"')]+extractsarcastic.identify_sentiment(text)
    def classify(self,features,target):
        print features,target
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(np.array(features), np.array(target))
        vector = self.extract_features_sentence("I love exams!! Can't wait for more.")

        print neigh.predict([vector])
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

