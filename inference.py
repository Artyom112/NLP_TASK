import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from joblib import load
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        return csr_matrix(X).todense()


class Classifier:
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_data(self, data):
        cleaned_sent = []
        for sent in data:
            sent = sent.lower()  # lower string
            sent = re.sub(r'\d+', '', sent)  # remove numbers
            sent = ' '.join(
                [word for word in sent.split() if word not in stopwords.words('english')])  # remove stop words
            sent = ' '.join(self.tokenizer.tokenize(sent))  # tokenize
            sent = re.sub(r'[^\w\s]', '', sent)  # remove punctuation
            sent = ' '.join([self.lemmatizer.lemmatize(word) for word in sent.split()])  # lemmatize words
            cleaned_sent.append(sent)
        return cleaned_sent

    def predict(self, dataframe, classifier_object):
        dataframe['cleaned_text'] = self.clean_data(dataframe['text'].values)
        predictions = classifier_object.predict(dataframe['cleaned_text'].values)
        return predictions


if __name__ == '__main__':
    classifier = Classifier()
    obj = load('/Users/artyomkholodkov/Downloads/models/model.pkl')
    frame = pd.read_csv('/Users/artyomkholodkov/Downloads/raw_text.csv')
    print(classifier.predict(frame.loc[:10, :], obj))
