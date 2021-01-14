from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model

def odqa_deeppavlov(questions):
    odqa = build_model(configs.odqa.en_odqa_infer_wiki, download = True)
    results = odqa(questions)
    return results

if __name__=="__main__":
    questions = ["Where did guinea pigs originate?", "Who is virat kohli?"]
    answers = odqa_deeppavlov(questions)
    print(answers)
