import nltk
import pickle
import pandas as pd
from textblob.classifiers import DecisionTreeClassifier
from textblob import TextBlob

def read_train_data(filename):

    xl = pd.read_excel(filename,"Twitter data")
    twitter = xl.drop(xl.columns[[1,2,3]],axis=1)
    twitter['SASI'] = twitter['SASI'].apply(lambda x: True if x >= 3 else False)
    xl = pd.read_excel(filename,"Amazon Data")
    amazon = xl.drop(xl.columns[[1,2,3]],axis=1)
    amazon['SASI'] = amazon['SASI'].apply(lambda x: True if x >= 3 else False)
    concatenated = pd.concat([amazon,twitter])
    featuresets = ([(Text, SASI) for index, (Text, SASI) in concatenated.iterrows()])
    return featuresets

def read_test_data(filename):
    df = pd.read_csv(filename)
    featuresets = df['text'] #[(Text, SASI) for index, (text) in df.iterrows() ]
    return featuresets
def setup_classifier(train,test):
    print ("setting up classifier -----")
    cl = DecisionTreeClassifier(train)
    pickle.dump(cl,open("classifier.p","wb"))
    print("Classifier object dumped -------")

def classify(test):
    cl = pickle.load(open("classifier.p","rb"))
    print(test.shape)
    count=0
    for each_review in test:
        try:
            print(count)
            blob = TextBlob(each_review, classifier=cl)
            result=blob.classify()
            """ for sentence in blob.sentences:
                print(sentence)
                print(sentence.classify())
                """
            if(result==True):
                print(blob)
                count=count+1
        except(UnicodeDecodeError):
            pass
    print(count)
def main():
    filename = "../data/annotated.xls"
    #train_features=read_train_data(filename)
    #pickle.dump(train_features,open( "train.p", "wb" ))
    #train_features=pickle.load(open("train.p","rb"))
    #print("Train features loaded from pickle")
    filename = "../data/ReviewsWithScore.csv"
    test=read_test_data(filename)
    print ("Extracted test data")
    #setup_classifier(train_features,test)
    print("Classifier setup")

    classify(test)

if __name__ == '__main__':
    main()