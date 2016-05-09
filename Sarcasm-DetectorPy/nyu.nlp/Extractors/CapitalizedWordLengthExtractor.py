from sklearn.base import BaseEstimator, TransformerMixin

class CapitalizedWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def capitalized_words(self, name):
        count=0
        for word in name.split():
            if self.isCapital(word)==True:
                count=count+1
        return count
    def isCapital(self,str):
        if str.isupper():
            return True
        else :
            return False
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df['Text'].apply(self.capitalized_words)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self