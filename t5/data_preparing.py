import pandas as pd


def main():
    with open("""C:/Users/User/Desktop/Programming/classification/t5/dialogs.txt""", encoding='utf8') as f:
        lines = [line.rstrip() for line in f]
    
    print(lines[:20])
    questions = list()
    answers = list()
    flag = 0
    for line in lines:
        if not line:
            flag = 0
        if line and flag == 0:
            if len(questions) == len(answers):
                questions.append(line)
            else:
                answers.append(line)
                flag = 1

    df = pd.DataFrame({'Questions': questions, 'Answers': answers})

    df.to_excel('./t5/data3.xlsx')

if __name__=='__main__':
    main()

