from flask import Flask, request
import json
from QuestionAnsweringSystem.QuestionAnswer import get_answer_using_bert
app=Flask(__name__)

@app.route ("/questionAnswering", methods=['POST'])
def questionAnswering():
    try:
        json_data = request.get_json(force=True)
        query = json_data['query']
        context_list = json_data['context_list']
        result = []
        for val in context_list:
            context = val['context']
            context = context.replace("\n"," ")
            answer_json_final = dict()
            answer = get_answer_using_bert(context, query)
            answer_json_final['answer'] = answer
            answer_json_final['id'] = val['id']
            answer_json_final['question'] = query
            result.append(answer_json_final)

        result={'results':result}
        result = json.dumps(result)
        return result

    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__" :
    app.run(debug=True,port="5001")
