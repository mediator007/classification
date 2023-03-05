import joblib


def prediction(question):
    pipeline = joblib.load('pipline.pkl')
    answer = pipeline.predict([question])[0]
    return answer

if __name__=='__main__':
    print('Answer --------- :', prediction('hi'))
    print('Answer --------- :', prediction('зачем а главное'))
    print('Answer --------- :', prediction('как заказать услугу и заключить договор'))