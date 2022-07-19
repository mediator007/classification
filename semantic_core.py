import sys
import numpy as np
import re
import nltk.stem as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

def text_cleaner(text):
    text = text.lower()
    stemmer = Stemmer('russian')
    text = ' '.join( stemmer.stemWords(text.split() ) )
    text = re.sub( r'\b\d+\b', ' digit ', text )
    return text

def load_data():
    data = { 'text':[],'tag':[] }
    for line in open('learning1.txt'):
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
    return {
        'train': { 'x': X[:-
nb_validation_samples], 'y': Y[:-
nb_validation_samples] },
        'text': {'x': X[-
nb_validation_samples:], 'y': Y[-
nb_validation_samples:] }
    }
    

def openai():
    data = load_data()
    D = train_test_split(data)

    text_clf = Pipeline([
        ('tfidf',TfidfVectorizer(use_idf=True)), # try stopwords = ..., vocabulary = ... .
        ('clf',SGDClassifier(loss='hinge')), # try parametres
        ])
    text_clf.fit(D['train']['x'], D['train']['x'] )

    z=input('Введите слова через запятую и пробел: ')
    zz=[]
    zz.append(z)
    predicted = text_clf.predict(zz)
    print(predicted[0])

if __name__ == '__main__':
    sys.exit(openai())