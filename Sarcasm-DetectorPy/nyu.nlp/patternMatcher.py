import csv
import nltk
import re
import statistics
from string import punctuation
class load_csv(object):
    def __init__(self):
        count = 0
        filename="../data/test.csv";
        text = []
        sarcastic_pats = self.process_amazon("../data/amazon.csv")
        with open(filename,"rb") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                reviewId = row['review_id']
                para = row['text']
                res = re.sub(r'(?<=['+punctuation+'])\s+(?=[A-Z])', '\n', para)
                res_sents = res.rstrip().splitlines()
                text.append(res_sents)
                count+= 1
                if count == 5:
                    break
        self.calculate_matches(text,sarcastic_pats)
        self.extract_punct(filename)
    def calculate_matches(self,text,sarcastic_pats):
        for para in text:
            for sent in para:
                sentpat = self.generate_patterns(sent)
                maxMatch = 0
                for sarcpat in sarcastic_pats:
                    if sarcpat == sentpat:
                        print "Exact Match" , sarcpat, sentpat, 1.0
                        break
                    else:
                        score = self.lcs(sarcpat,sentpat)
                        if float(score)/len(sarcpat) == 1.0:
                            print "Sparse Match", sarcpat, sentpat, 0.1
                            break

                        if maxMatch < score:
                            maxMatch = score
                            maxPat = sarcpat
                if maxMatch > 0:
                    print "Partial Match" , maxPat, sentpat, 0.1*float(maxMatch)/len(maxPat)
                else:
                    print "No Match", sentpat, 0
    def process_amazon(self,filename):
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

    def generate_all_patterns(self,reviews,cwset,hfwset):
        extracted_patterns=[]
        for each_review in reviews:
            tokens=nltk.word_tokenize(each_review)
            pattern=[]
            for each_token in tokens:
                isCW= cwset.__contains__(each_token)
                isHFW=hfwset.__contains__(each_token)
                pat = ''
                if isCW and isHFW:
                    pat = pat+'H'
                else :
                    if isCW:
                        pat=pat + 'C'
                    else :
                        pat=pat +'H'

            pattern.append(pat)
            extracted_patterns.append(pattern)
        return extracted_patterns

    def generate_patterns(self,sent):
        text = nltk.word_tokenize(sent)
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
new_class = load_csv()