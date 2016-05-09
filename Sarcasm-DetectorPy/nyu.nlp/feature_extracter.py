
import nltk
import extractsarcastic

class FeatureExtractor():

    def count_apost(self,text):
        return text.count('!')
    def count_qn(self,text):
        return text.count('?')
    def data_len(self,text):
        return len(text)
    def count_quotes(self,text):
        return text.count('\"')
    def extract_feature_dict(self,data,isTest=False):
        feature={}

        feature['!Count']=data['Text'].count('!')
        feature['?Count']=data['Text'].count('?')
        feature['Length']=len(data['Text'])
        feature['Score']=float(data['SASI'])
        feature['QuotesCount']=data['Text'].count('\"')
        feature['Sentiment']=extractsarcastic.identify_sentiment(str(data['Text']))

        return feature

    def extract_all_features(self,trainDataFrame,testDataFrame):
        train=[]
        test=[]

        train = ([(self.extract_feature_dict(row), row['SASI']) for index, row in trainDataFrame.iterrows()])
        print "++++++ Extracted TRAIN features"

        test= ([(row['Text']) for index, row in testDataFrame.iterrows()])
        print "++++++ Extracted TEST features"
        return train,test


