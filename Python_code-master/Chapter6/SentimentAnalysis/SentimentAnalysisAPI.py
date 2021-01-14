from flask import Flask, request
import json

from SentimentAnalysis.SentimentAnalysis import build_sentiment_model

app=Flask(__name__)

@app.route ("/sentimentAnalysis", methods=['POST'])
def sentimentAnalysis():
    try:
        json_data = request.get_json(force=True)
        questions = json_data['questions']
        sentiment_model = build_sentiment_model()
        questions_list =[]
        for ques in questions:
            questions_list.append(ques)

        model_output = sentiment_model(questions_list)
        index = 0
        result = []
        for ans in model_output:
            sentiment_qa =dict()
            sentiment_qa['qustion'] = questions_list[index]
            sentiment_qa['answer'] = ans
            result.append(sentiment_qa)

        result={'results':result}
        result = json.dumps(result)
        return result
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__" :
    app.run(port="5000")
