import joblib


def prediction(question):
    pipeline = joblib.load('pipline.pkl')
    answer = pipeline.predict([question])[0]
    return answer