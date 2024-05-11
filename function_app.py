import azure.functions as func
import logging

from data_3 import dfs
from model import model_train
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="prediction")
def prediction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    gener = req_body.get('gener')
    age = req_body.get('age')
    
    response = {key : model_train(values, key, age, gener) for key, values in dfs.items()}

    if gener and age:
        return func.HttpResponse(json.dumps(response), content_type='application/json')
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a age and genner in the query string or in the request body for a personalized response.",
             status_code=200
        )