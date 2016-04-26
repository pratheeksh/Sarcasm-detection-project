import pandas as pd
import numpy as np
from textblob import TextBlob


def extract_negative_reviews(filename):
    df=pd.read_csv(filename)
    neg_df=df[df['stars'] < 3]
    identify_positive_sentiment(neg_df)

def identify_positive_sentiment(neg_df):
    n = neg_df.shape[0]
    i = 0
    review_arr = np.zeros(n, dtype=np.float32)
    for eachrow in neg_df.itertuples():
        text = eachrow[3]
        blob = TextBlob(text)
        sentimentscore = 0
        try:
            for sentence in blob.sentences:
                sentimentscore = sentimentscore + sentence.sentiment.polarity
            sentimentscore = sentimentscore/blob.sentences.__sizeof__()
            print(text,sentimentscore)
        except (RuntimeError, TypeError, NameError,UnicodeDecodeError):
            pass
        review_arr[0] = sentimentscore
        i=i+1
    neg_df.insert(0,'Score',review_arr)
    neg_df.to_csv("ReviewsWithScore.csv")

def main():
    filename="/Users/bharathipriyaa/Desktop/NLP-Project/Sarcasm-DetectorPy/data/aaaa-yelp_academic_dataset_review.csv"
    extract_negative_reviews(filename)
if __name__ == '__main__':
    main()