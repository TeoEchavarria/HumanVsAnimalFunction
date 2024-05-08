import azure.functions as func
import logging
import numpy as np
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="human_vs_animal")
def human_vs_animal(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    x = np.linspace(0, 15, 50)
    y = np.sin(x)

    # Convertir los datos a una lista de tuplas o a otro formato que prefiera el frontend
    data = list(zip(x, y))

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(
        json.dumps(data),
        mimetype="application/json"
    )
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )