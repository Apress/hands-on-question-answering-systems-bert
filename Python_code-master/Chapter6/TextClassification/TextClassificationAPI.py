from flask import Flask, request
import json

from TextClassification.TextClassification import create_text_classification_model, predict_category
from TextClassification import create_text_classification_model

app=Flask(__name__)


@app.route ("/textClassification", methods=['POST'])
def textClassification ():
    try:
        json_data = request.get_json(force=True)
        input_text = json_data['query']

        classification_model, trans = create_text_classification_model()
        category = predict_category(classification_model, trans, input_text)

        result = {}
        result['query'] = input_text
        result['category'] = category

        result = json.dumps(result)
        return result

    except Exception as e:
        error = {"Error": str(e)}
        error = json.dumps(error)
        return error


if __name__ == "__main__" :
    app.run(port="5000")
