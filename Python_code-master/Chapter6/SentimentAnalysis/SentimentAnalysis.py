from deeppavlov import build_model, configs

def build_sentiment_model():
    model = build_model(configs.classifiers.insults_kaggle_bert, download=True)
    return model


if __name__=="__main__":

    test_input = ['This movie is good.', 'You are so dumb!']
    sentiment_model = build_sentiment_model()
    results = sentiment_model(test_input)
    print(results)
