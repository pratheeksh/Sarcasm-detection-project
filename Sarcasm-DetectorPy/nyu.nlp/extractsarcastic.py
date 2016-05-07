import pandas as pd
import numpy as np
from textblob import TextBlob


def identify_sentiment(text):
    blob = TextBlob(text)
    sentimentscore = 0
    try:
        return list(blob.sentiment)

    except (RuntimeError, TypeError, NameError,UnicodeDecodeError):
        print "this is fucking up",blob

def main():
    print identify_sentiment("Simple is better than complex")
if __name__ == '__main__':
    main()
