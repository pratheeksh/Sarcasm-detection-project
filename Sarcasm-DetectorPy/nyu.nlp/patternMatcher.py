import csv
import nltk
import re
import statistics
from string import punctuation
class load_csv():


    def init(self,cwset,hfwset):
        print "hello bharathi"
        count = 0
        self.cwset=cwset
        self.hfwset=hfwset
        filename="../data/test.csv";
        text = []
        sarcastic_pats = self.process_amazon("../data/amazon.csv",cwset,hfwset)
        output=[]
        res = self.generate_sents(filename)
        first_count = 0
        second_count = 0
        test_data = []
        for tup in res:
            res_sents = tup[0]
            reviewId = tup[1]
            for each_sentence in res_sents:
                first_count+= 1
                second_count+= 1
                if second_count > 500:
                    break
                if each_sentence is not None:
                    temp_dict={}

                    score=self.calculate_matches(each_sentence,sarcastic_pats)
                    temp_dict['Text'] = each_sentence
                    temp_dict['Review_id'] = "{}{}".format(reviewId,count)
                    temp_dict['Score'] = score
                    if first_count < 200:
                        test_data.append(temp_dict)
                    else:
                        output.append(temp_dict)
        return output,test_data
    def generate_sents(self,filename):
        output = [] 
        with open(filename,"rb") as csvfile:
            reader = csv.DictReader(csvfile)
            count=0
            for row in reader:
                if(count>100):
                    break;
                count+=1;
                reviewId = row['review_id']
                para = row['text']
                res = re.sub(r'(?<=['+punctuation+'])\s+(?=[A-Z])', '\n', para)
                res_sents = res.rstrip().splitlines()
                output.append((res_sents,reviewId))
            return output

    def calculate_matches(self,text,sarcastic_pats):

        sentpat = self.generate_patterns(text)
        maxMatch = 0
        for sarcpat in sarcastic_pats:
            if sarcpat == sentpat:
                #print "Exact Match" , sarcpat, sentpat, 1.0
                break
            else:
                score = self.lcs(sarcpat,sentpat)
                if float(score)/len(sarcpat) == 1.0:
                   # print "Sparse Match", sarcpat, sentpat, 0.1
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
                b, c, d, e = float(row['MT1']), float(row['MT2']), float(row['MT3']), float(row['SASI'])
                avg_score = statistics.mean([b,c,d,e])
                if avg_score >= 3:
                    pat = self.generate_patterns(text)
                    if pat not in sarcastic_patterns:
                        sarcastic_patterns.append(pat)
        return sarcastic_patterns

    def generate_patterns(self,text):
        pattern=[]
        if(text is None):
            return None

        tokens=nltk.word_tokenize(text.decode("utf-8"))
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
        return pattern



    def extract_punct(self,filename):
        test_data=[]
        with open(filename,"rb") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    text = row['text']
                test_data.append(text)
        featuresets = [(self.extract_features(sentence) for sentence in test_data)]


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
