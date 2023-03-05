import joblib


def prediction(question):
    pipeline = joblib.load('pipeline.pkl')
    answer = pipeline.predict([question])[0]
    return answer

if __name__=='__main__':
    ...
    