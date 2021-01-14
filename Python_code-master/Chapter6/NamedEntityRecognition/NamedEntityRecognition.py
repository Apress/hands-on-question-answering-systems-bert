from deeppavlov import build_model, configs
import pandas as pd
def build_ner_model ():
    model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
    return model

if __name__=="__main__": 
    test_input = ['Amazon rainforests are located in South America.']
    ner_model = build_ner_model()
    results = ner_model(test_input)
    print(results[0][0])
    print(results[1][0])
    results = pd.DataFrame(zip(results[0][0],results[1][0]), columns=['Word','Entity'])
    print(results)
