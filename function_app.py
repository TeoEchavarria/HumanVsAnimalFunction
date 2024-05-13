import azure.functions as func
import logging

from model import model_train
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="prediction")
def prediction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    gener = req.params.get('gener')
    age = req.params.get('age')
    if not gener and not age:
        try:
            req_body = req.get_json()
        except ValueError:
            raise "Not parameters"
        else:
            gener = req_body.get('gener')
            age = req_body.get('age')
    
    try:
        keys = ["rat", "house cat", "medium dog", "large dog", "kangaroo", "eagle", "grizzly bear", "wolf", "lion", "gorilla", "chimpanzee", "king cobra", "elephant", "crocodile", "goose"]
        response = {key : model_train(age, gener, key) for key in keys}
        response_body = json.dumps(response)
    except:
        raise "Prediction Failed"

    if gener and age:
        return func.HttpResponse(response_body)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a age and genner in the query string or in the request body for a personalized response.",
             status_code=200
        )