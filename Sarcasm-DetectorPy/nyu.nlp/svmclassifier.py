
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import feature_extracter as fe
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import extractsarcastic
class SVMClassify():

    def init(self):
        print "Classification through SVM"

        train,target,test=self.load_data()
        return train,target,test

    def load_data(self):

        train=pd.read_csv("../data/amazon.csv")
        train['Score']= train['Score'].apply(lambda x: 1 if x >3 else 0)
        test=train[100:]
        # reddit=pd.read_csv("../data/reddit.csv")
        #test=pd.read_csv("../data/reddit.csv") #train[101:]
        train=train[:100]
        #test['Score']= test['SASI'].apply(lambda x: 1 if x =="yes" else 0)
        return train,train['Score'],test


    def extract_features(self,traindata,testdata):

        train,target,test = [],[],[]
        for index,data in testdata.iterrows():
            test.append(self.extract_features_sentence(data['Text']))
        for index,data in traindata.iterrows():
            train.append(self.extract_features_sentence(data['Text']))
            target.append(data['Score'])
        return train,target,test

    def extract_features_sentence(self,data):
        fv=fe.FeatureExtractor()

        feature_vector=[ fv.count_apost(data),
                         fv.count_qn(data),
                         fv.data_len(data),
                         fv.count_quotes(data),
                         fv.sarcastic_score(data)]

        return feature_vector

    def classify(self,train,target,test):
        #print features,target
        output = []
        neigh = KNeighborsClassifier(n_neighbors=6)
        neigh.fit(np.array(train), np.array(target))
        for vector in test:
            output.extend(neigh.predict([vector]))
        return output

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks =[0,1]
        plt.xticks(tick_marks, ["Sarcastic","Non-Sarcastic"], rotation=45)
        plt.yticks(tick_marks, ["Sarcastic","Non-Sarcastic"])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


def main():
    svm=SVMClassify()
    train_data,target_data,test_data=svm.init()

    print "Extracted features sets for train and set. "
    print "Setting up the classifier"
    train,target,test=svm.extract_features(train_data,test_data)
    predicted=svm.classify(train,target,test)

    for each_test,prediction,actual in zip(test_data['Text'],predicted,test_data['Score']):
        print "Sentence {} with Sarcasm predicted as= {} while actual is={}".format(each_test,prediction,actual)
    print precision_recall_fscore_support(test_data['Score'], predicted,average='binary')
    print confusion_matrix(test_data['Score'], predicted)
    plt.show(block=True)
if __name__ == '__main__':
    main()
