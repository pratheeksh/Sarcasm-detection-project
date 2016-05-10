import feature_extracter as fe
import knnclassifier as kc
import pandas as pd
from nltk.tokenize import sent_tokenize
import patternExtract as pe
import patternMatcher as patternMatcher
import topics as topic
import evaluationMetrics as evalMet
class SarcasmClassifier():
    def classify_reddit(self):
        """ One time code to split reddit reviews

        df=pd.read_csv("../data/reddit.csv")
        train_df=df.copy(True)
        train_df['Score']= df['SASI'].apply(lambda x: 1 if x=="yes" else 0)
        train_df['True']=df['SASI'].apply(lambda x: 1 if x=="yes" else 0)
        train_df=self.split_reddit_reviews(train_df)
        """
        #Load files
        train_df=pd.read_csv("../data/reddit_reviews.csv")
        test_df=train_df[100:]
        test_df['Actual']= test_df['Score']
        print "Loading   reddit reviews for Sarcasm detection"


        print "Loading LDA model and Unigram Dict for topic detection"
        #Load LDS model for topics
        tc=topic.Topic()
        feat = fe.FeatureExtractor()
        feat.generateUnigramVectorizer(train_df['Text'].values.tolist())
        tc.getLdaModel(train_df['Text'].values.tolist())


        # Pattern Extraction -
        print "+++++++++ Extracting patterns +++++++++"
        pattext = pe.PatternExtraction(test_df['Text'])
        dic = pattext.calculateCorpusFrequency()
        cwset = pattext.findCW(dic)
        hfwset=pattext.findHFW(dic)

        print "+++++++++ Pattern matching +++++++++"
        # Pattern Matching -
        pm =patternMatcher.load_csv()
        pm.init(cwset,hfwset,"../data/reddit_reviews.csv","reddit")
        train_features,test_features=pm.getScoreFeatures(train_df),pm.getScoreFeatures(test_df) #scores_features[:500],scores_features[501:]

        print "++++++++ KNN Classification starts here  +++++++"
        knnobj=kc.knnClassifier()
        train,target,test = knnobj.extract_features_train(train_features,test_features,tc,feat)
        predicted = knnobj.classify(train,target,test)

        print "++++++++ Evaluation starts here ++++++++++"

        for (index,each_sentence),prediction in zip(test_df.iterrows(),predicted):
            print "KNN-Sentence: {} with actual sarcasm={} got predicted as {}".format(each_sentence['Text'],each_sentence['Actual'],prediction)
        ev = evalMet.Evaluation()
        ev.evaluate_results(expected=test_df['Actual'].tolist(),predicted=predicted)
    def load_sarcastic_alone(self):
        """
        One time module to looad just the sarcastic reviews
        :return:
        """
        train_df=pd.read_csv("reddit_reviews.csv")
        train_df=train_df[train_df['Score']==1]
        train_df.to_csv("reddit_sarcastic.csv")


    def split_reddit_reviews(self,reviews):
        columns = ['Text','Score', 'True']
        #Calculate total number of sentences to fill up the data frame
        count=0
        for index,each_review in reviews.iterrows():

            split_sentences=sent_tokenize(each_review['Text'])
            count+=len(split_sentences)
        print "total number of sentences {}".format(count)

        df = pd.DataFrame(index=range(0,count), columns=columns)
        Text,Score,True=[],[],[]
        for index,each_review in reviews.iterrows():
            split_sentences=sent_tokenize(each_review['Text'])
            actual_tag=each_review['True']
            score_tag=each_review['Score']
            for each_split_sentence in split_sentences:
                Text.append(each_split_sentence)
                Score.append(actual_tag)
                True.append(score_tag)
        print "Count ={} Text.length {}".format(count,len(Text))
        df['Text']=Text
        df['Score']=Score
        df['True']=True
        df.to_csv('reddit_reviews.csv')

    def classify(self):
        #Load files
        train_df=pd.read_csv("../data/amazon.csv")
        train_df['True']=train_df['Score']
        train_df['Score']= train_df['Score'].apply(lambda x: 1 if x >3 else 0)
        test_df=train_df[100:]
        test_df['Actual']= test_df['Score']
        test_file="../data/reddit.csv"
        train_file="../data/amazon.csv"
        print "Loading Amazon and reddit reviews for Sarcasm detection"
        #reviews=self.load_data(test_file)
        #amazon=self.load_data(train_file)

        print "Loading LDA model for topic detection"
        #Load LDS model for topics
        tc=topic.Topic()
        feat = fe.FeatureExtractor()
        feat.generateUnigramVectorizer(train_df['Text'].values.tolist())
        tc.getLdaModel(train_df['Text'].values.tolist())


        # Pattern Extraction -
        print "+++++++++ Extracting patterns +++++++++"
        pattext = pe.PatternExtraction(test_df['Text'])
        dic = pattext.calculateCorpusFrequency()
        cwset = pattext.findCW(dic)
        hfwset=pattext.findHFW(dic)

        print "+++++++++ Pattern matching +++++++++"
        # Pattern Matching -
        pm =patternMatcher.load_csv()

        #scores_features = [Text,ReviewId,Score]
        #scores_features=pm.init(cwset,hfwset,test_file)
       # amazon_test=amazon[100:]
       # amazon_test['Score']=amazon_test['Score'].apply(lambda x: 1 if x >3 else 0)
        # train_features and test_features are data frames
        #scores_features=pm.init(cwset,hfwset)
        train_features,test_features=pm.getScoreFeatures(train_df),pm.getScoreFeatures(test_df) #scores_features[:500],scores_features[501:]
        #train_features,test_features=,amazon_test

        print "++++++++ KNN Classification starts here  +++++++"
        knnobj=kc.knnClassifier()
        train,target,test = knnobj.extract_features_train(train_features,test_features,tc,feat)
        predicted = knnobj.classify(train,target,test)

        print "++++++++ Evaluation starts here ++++++++++"

        for (index,each_sentence),prediction in zip(test_df.iterrows(),predicted):
            print "KNN-Sentence: {} with actual sarcasm={} got predicted as {}".format(each_sentence['Text'],each_sentence['Actual'],prediction)
        ev = evalMet.Evaluation()
        ev.evaluate_results(expected=test_df['Actual'].tolist(),predicted=predicted)


        #Modified KNN
        #ev.evaluate(twitter_out,expected)
        """
        mknnObj=mKNN.ModifiedKNN()
        predictions=mknnObj.predict_test(train,train_df['True'],test)
        for (index,each_sentence),prediction in zip(test_df.iterrows(),predictions):
            print "Modified KNN-Sentence: {} with actual sarcasm={} got predicted as {}".format(each_sentence['Text'],each_sentence['Actual'],prediction)
        ev = evalMet.Evaluation()
        ev.evaluate_results(expected=test_df['Actual'].tolist(),predicted=predictions)
        """
    def load_data(self,filename):
        df=pd.read_csv(filename)
        df['Text']=df['Text'].apply(lambda text: text.decode("utf-8"))
        return df
    def normalize_reddit(self,filename):
        df=pd.read_csv(filename)

        for index,row in df.iterrows():
            sentences=row['Text']


def main():
    sc=SarcasmClassifier()
    #sc.classify()
    #sc.classify_reddit()
    #sc.load_sarcastic_alone()
    sc.classify_reddit()
if __name__ == '__main__':
    main()
