from operator import index
import sys
import numpy as np
import re
import nltk.stem as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics, svm
import joblib

def text_cleaner(text):
    text = text.lower()
    stemmer = Stemmer('russian')
    text = ' '.join( stemmer.stemWords(text.split() ) )
    text = re.sub( r'\b\d+\b', ' digit ', text )
    return text

def load_data():
    data = { 'text':[],'tag':[] }
    for line in open('learning.txt', encoding='utf-8'):
        if(not('#' in line)):
            row = line.split("@")
            data['text'] += [row[0]]
            data['tag'] += [row[1]]
    return data

def train_test_split( data, validation_split = 0.1):
    sz = len(data['text'])
    indices = np.arange(sz)
    np.random.shuffle(indices)

    X = [ data['text'][i] for i in indices ]
    Y = [ data['tag'][i] for i in indices ]
    nb_validation_samples = int(validation_split * sz)
    result =  {
        'train': {
            'x': X[:-nb_validation_samples], 
            'y': Y[:-nb_validation_samples]
            },
        'test': {
            'x': X[-nb_validation_samples:], 
            'y': Y[-nb_validation_samples:]
            },
        'all': {
            'x': X[:], 
            'y': Y[:]
            },
    }
    return result
    

def get_prediction():
    data = load_data()
    D = train_test_split(data)

    pipeline = Pipeline([
        ('tfidf',TfidfVectorizer(use_idf=True)), # try stopwords = ..., vocabulary = ... .
        ('clf',SGDClassifier(loss='hinge')), # try parametres
        ])
    pipeline.fit(D['all']['x'], D['all']['y'])

    joblib.dump(pipeline, 'pipeline.pkl', compress = 1)
    pipeline = joblib.load('pipeline.pkl')

    # check pipeline
    predict = pipeline.predict(["заказать iaas"])[0]
    print(predict)

    # https://habr.com/ru/post/538458/
    # predicted_sgd = pipeline.predict(D['test']['x'])
    # print(metrics.classification_report(predicted_sgd, D['test']['y']))

if __name__ == '__main__':
    sys.exit(get_prediction())