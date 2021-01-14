from flask import Flask, request
import json
from DeeppavlovQASystem.QA_Deepplavlov import qa_deeppavlov

app=Flask(__name__)

@app.route ("/qaDeepPavlov", methods=['POST'])
def qaDeepPavlov():
    try:
        json_data = request.get_json(force=True)
        query = json_data['query']
        context_list = json_data['context_list']
        result = []
        for val in context_list:
            context = val['context']
            context = context.replace("\n"," ")
            answer_json_final = dict()
            answer = qa_deeppavlov(context, query)
            answer_json_final['answer'] = answer
            answer_json_final['id'] = val['id']
            answer_json_final['question'] = query
            result.append(answer_json_final)

        result = json.dumps(result)
        return result
    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__" :
    app.run(port="5000")
