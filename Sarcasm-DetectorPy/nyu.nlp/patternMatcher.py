import csv
import nltk
import re
import statistics
import pandas as pd
from string import punctuation
class load_csv():


    def init(self,cwset,hfwset,filename="",dataset=""):

        self.cwset=cwset
        self.hfwset=hfwset
        text = []
        if dataset=="amazon":
            sarcastic_pats = self.process_amazon("../data/amazon.csv",cwset,hfwset)
        else :
            sarcastic_pats = self.process_reddit("reddit_sarcastic.csv",cwset,hfwset)
        res = pd.read_csv(filename)
        test_data = []
        for index,row in res.iterrows():
            each_sentence,funnyScore = row['Text'],row['Score']

            if len(each_sentence)!=0:
                 row['Score']=self.calculate_matches(each_sentence,sarcastic_pats)

        res['Score']=res['Score'].apply(lambda x: 1 if x >3.0 else 0)
        return res

    def getScoreFeatures(self,df):
        text = []
        sarcastic_pats = self.process_amazon("../data/amazon.csv",None,None)
        for index,row in df.iterrows():
            each_sentence,Score = row['Text'],row['Score']
            if len(each_sentence)!=0:
                 row['Score']=self.calculate_matches(each_sentence,sarcastic_pats)

        #res['Score']=res['Score'].apply(lambda x: 1 if x >3.0 else 0)
        return df

    def match_test_patterns(self,cwset,hfwset):
        filename="../data/Twitter.csv";
        text = []
        sarcastic_pats = self.process_amazon("../data/amazon.csv",cwset,hfwset)
        output=[]
        first_count = 0
        second_count = 0
        test_data = []
        expected = []
        tweets = []
        with open(filename,"rb") as csvfile:
            reader = csv.DictReader(csvfile)
            count=0
            output = []
            for row in reader:
                if(count>100):
                    break;
                count+=1;
                reviewId = count 
                para = row['TWEET']

                b, c, d, e = float(row['MT1']), float(row['MT2']), float(row['MT3']), float(row['SASI'])
                avg_score = statistics.mean([b,c,d,e])
                expected.append((reviewId,avg_score))
                res = re.sub(r'(?<=['+punctuation+'])\s+(?=[A-Z])', '\n', para)
                res_sents = res.rstrip().splitlines()

                tweets.append((res_sents,reviewId))

            for tup in tweets:
                res_sents = tup[0]
                reviewId = tup[1]
                funnyScore = 4
                for each_sentence in res_sents:
                   if each_sentence is not None:
                       temp_dict={}
                       score=self.calculate_matches(each_sentence,sarcastic_pats)
                       temp_dict['Text'] = each_sentence
                       temp_dict['Review_id'] = "{}".format(reviewId)
                       temp_dict['Score'] = score
                       temp_dict['Funny Score'] = "{}".format(funnyScore)
                       output.append(temp_dict)
            return output,expected

    def generate_sents(self,filename):
        output = [] 
        expected = []
        with open(filename,"rb") as csvfile:
            reader = csv.DictReader(csvfile)
            count=0
            for row in reader:
                if(count>100):
                    break;
                count+=1;
                reviewId,para,funnyScore,bussID,coolScore = row['review_id'],row['Text'],row['votes.funny'],row['business_id'],row['votes.cool']
                res = re.sub(r'(?<=['+punctuation+'])\s+(?=[A-Z])', '\n', para)
                res_sents = res.rstrip().splitlines()
                output.append((res_sents,reviewId,funnyScore))
            return output

    def calculate_matches(self,text,sarcastic_pats):

        sentpat = self.generate_patterns(text)
        maxMatch = 0
        for sarcpat in sarcastic_pats:
            if sarcpat == sentpat:
                break
            else:
                score = self.lcs(sarcpat,sentpat)
                if float(score)/len(sarcpat) == 1.0:
                    break
                if maxMatch < score:
                    maxMatch = score
                    maxPat = sarcpat
        if maxMatch > 0:
            return 0.1*float(maxMatch)/len(maxPat)
        else:
            return 0
    def process_amazon(self,filename,cwset,hfwset):
        sarcastic_patterns = []
        with open(filename,"rb") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row['Text']
                b, c, d, e = float(row['MT1']), float(row['MT2']), float(row['MT3']), float(row['Score'])
                avg_score = statistics.mean([b,c,d,e])
                if avg_score > 3:
                    pat = self.generate_patterns(text)
                    if pat not in sarcastic_patterns:
                        sarcastic_patterns.append(pat)
        return sarcastic_patterns
    def process_reddit(self,filename,cwset,hfwset):
        sarcastic_patterns = []
        sarcastic_reviews=pd.read_csv(filename)
        for index,row in sarcastic_reviews.iterrows():
            text = row['Text']
            pat = self.generate_patterns(text)
            if pat not in sarcastic_patterns:
                sarcastic_patterns.append(pat)
        return sarcastic_patterns
    def generate_patterns(self,text):
        pattern=[]
        if(text is None):
            return None

        text=nltk.word_tokenize(text.decode("utf-8"))
        tagged_sent = nltk.pos_tag(text)
        noun = re.compile('NN|NNS')
        verb = re.compile('VB*')
        adj = re.compile('JJ*')
        adv = re.compile('RB*')
        res_list = []

        for tup in tagged_sent:
            word = tup[0]
            pos = tup[1]
            pat = ''
            res = bool(noun.match(pos))|bool(verb.match(pos))|bool(adj.match(pos))|bool(adv.match(pos))
            if res:
                new_tup = (word, "CW")
                pat = pat+'C'
            else:
                new_tup = (word, "HFW")
                pat = pat+'H'
            res_list.append(pat)
        return res_list
        """
        for each_token in tokens:
            isCW= self.cwset.__contains__(each_token)
            isHFW=self.hfwset.__contains__(each_token)
            pat = ''
            if isCW and isHFW:
                pat = pat+'H'
            else :
                if isCW:
                    pat=pat + 'C'
                else :
                    pat=pat +'H'

            pattern.append(pat)
        """
        return pattern




    def lcs(self,pattern,sent):
        pattern = ''.join(pattern)
        sent = ''.join(sent)
        memo = [ [ 0 for j in range(len(sent)+1) ] for i in range(len(pattern)+1) ]
        for i,x in enumerate(pattern):
            for j,y in enumerate(sent):
                if  x == y:
                    memo[i+1][j+1] = memo[i][j] + 1
                else:
                    memo[i+1][j+1] = max( memo[i][j+1], memo[i+1][j] )
        return memo[len(pattern)][len(sent)]
        #return memo
def main():
    print "In Pattern Matcher"
if __name__ == '__main__':
    main()
