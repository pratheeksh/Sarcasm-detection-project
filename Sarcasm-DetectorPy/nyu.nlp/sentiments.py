import nltk
import numpy as np
import string
import load_sent
from textblob import TextBlob

class Sentiment():
    def  calculateSent(self,sentence,sentiments):
        tokens = nltk.word_tokenize(sentence)
        tokens = [(t.lower()) for t in tokens]
        mean_sentiment = sentiments.score_sentence(tokens)
        if len(tokens) == 2:
            tokens.append(".")
        first_half = tokens[:len(tokens)/2]
        second_half = tokens[(len(tokens)/2):]
        mean_sent_f = sentiments.score_sentence(first_half)
        mean_sent_s = sentiments.score_sentence(second_half)
        avg_sent_f = mean_sent_f[0] - mean_sent_f[1]
        avg_sent_s = mean_sent_s[0] - mean_sent_s[1]
        first_t = tokens[:len(tokens)/3]
        second_t = tokens[len(tokens)/3:2*len(tokens)/3]
        third_t = tokens[2*len(tokens)/3:]
        mean_sent_ft = sentiments.score_sentence(first_t)
        mean_sent_st = sentiments.score_sentence(second_t)
        mean_sent_tt = sentiments.score_sentence(third_t)
        avg_sent_ft = mean_sent_ft[0] - mean_sent_ft[1]
        avg_sent_st = mean_sent_st[0] - mean_sent_st[1]
        avg_sent_tt = mean_sent_tt[0] - mean_sent_tt[1]
        res = [mean_sent_ft[0]]+[ mean_sent_ft[1]]+[mean_sent_st[0]]+[mean_sent_st[1]]+[mean_sent_tt[0]] + [mean_sent_tt[1]] + [avg_sent_ft] + [avg_sent_st] + [avg_sent_tt] + [abs(avg_sent_ft-avg_sent_tt)] 

        
        return [mean_sent_f[0], mean_sent_f[1], mean_sent_s[0], mean_sent_s[1], avg_sent_f, avg_sent_s, abs(avg_sent_s-avg_sent_f) ]+res
def main():
    p =  Sentiment ()
    print p.calculateSent("my life is horrible  but I love it anyway.")
if __name__ == '__main__':
    main()
