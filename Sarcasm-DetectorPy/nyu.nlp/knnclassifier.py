
from sklearn.neighbors import KNeighborsClassifier
class Classifier():
### Define specific features
    def extract_features(self,text):
        res = []
        for sent in text:
            res.append(self.extract_features_sentence(sent))
        
    def extract_features_sentence(self,text):
        print text
        return [ text.count('!'),
                 text.count('?'),
                 text.length,
                text.count('\"')]
    def classify(self,features,target):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(features, target)
        print neigh.predict(
