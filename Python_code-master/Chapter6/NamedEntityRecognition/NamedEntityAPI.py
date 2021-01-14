from flask import Flask, request
import json

from NamedEntityRecognition.NamedEntityRecognition import build_ner_model

app=Flask(__name__)

@app.route ("/namedEntity", methods=['POST'])
def namedEntity():
    try:
        json_data = request.get_json(force=True)
        query = json_data['query']
        ner_model = build_ner_model()
        model_output = ner_model([query])
        words= model_output[0][0]
        tags = model_output[1][0]
        result_json = dict()
        result_json['query'] = query
        entities = []
        index = 0

        for word in words:
            word_tag_dict=dict()
            word_tag_dict['word'] = word
            word_tag_dict['tag'] = tags[index]
            index = index+1
            entities.append(word_tag_dict)

        result_json['entities'] = entities
        result = json.dumps(result_json)
        return result
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__" :
    app.run(port="5000")
