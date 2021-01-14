from deeppavlov import build_model, configs

def qa_deeppavlov(question, context):
    model = build_model(configs.squad.squad_bert, download=True)
    result = model([context], [question])
    return result

if __name__=="__main__":

    context = " In 1888, The Football League was founded in England, becoming the first of many professional football competitions. During the 20th century, several of the various kinds of football grew to become some of the most popular team sports in the world."

    question = "In which year the Football league was founded?”

    answers = qa_deeppavlov(context, question)
    print(answers)
