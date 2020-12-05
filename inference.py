import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from joblib import load
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
import numpy as np


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        return csr_matrix(X).todense()


class Classifier:
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_data(self, dataframe):
        cleaned_sent = []
        for sent in dataframe:
            sent = sent.lower()
            sent = re.sub(r'\d+', '', sent)
            sent = ' '.join(
                [word for word in sent.split() if word not in stopwords.words('english')])
            sent = ' '.join(self.tokenizer.tokenize(sent))
            sent = re.sub(r'[^\w\s]', '', sent)
            sent = ' '.join([self.lemmatizer.lemmatize(word) for word in sent.split()])
            cleaned_sent.append(sent)
        return cleaned_sent

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def predict(self, dataframe, classifier_object):
        preds_all_texts = []
        dataframe['cleaned_text'] = self.clean_data(dataframe['text'].values)
        texts = dataframe['cleaned_text'].values
        for text in texts:
            pred_data = np.array([' '.join(part) for part in self.chunks(text.split(), 200)])
            preds = classifier_object.predict(pred_data)
            counts = np.bincount(preds)
            preds_all_texts.append(np.argmax(counts))
        return np.array(preds_all_texts)


if __name__ == '__main__':
    classifier = Classifier()
    obj = load('/Users/artyomkholodkov/Downloads/models/model.pkl')
    frame = pd.read_csv('/Users/artyomkholodkov/Downloads/raw_text.csv')
    print(classifier.predict(frame.loc[:10, :], obj))
