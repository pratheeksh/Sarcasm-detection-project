import pandas as pd
import numpy as np
from textblob import TextBlob


def identify_sentiment(text):
    blob = TextBlob(text)
    sentimentscore = 0
    try:
        return blob.sentiment[0]
    except (RuntimeError, TypeError, NameError,UnicodeDecodeError):
        return 0

def main():
    print identify_sentiment("Simple is better than complex")
if __name__ == '__main__':
    main()
