

class Classifier():
### Define specific features

    def extract_features(self,text):
        print text
        return {'number_apost': text.count('!'),
                'number_exclam':text.count('?'),
                'sentence_length': text.length,
                'number_quotes':text.count('\"')}