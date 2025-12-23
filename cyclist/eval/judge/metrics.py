import re
import json
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_count_metrics(results):
    """
    Compute count metrics from the results of the judge.
    Args:
        results (list): List of results from the judge, each result is a dictionary with keys 'custom_id' and 'response'.
    """
    score = 0
    evaluated = 0
    y_true = []
    y_pred = []
    
    
    for result in results:
        # print(f"Request {result['custom_id']}:")
        # print(f"Response: {result['response']}")
        response = result['response']['body']['choices']['message']['content']

        try:
            json_response = re.search(r'{(.|\r?\n)*}', response).group()
            json_response = json.loads(json_response)
        
            score += json_response['score']
            abs_error = np.abs(float(json_response['prediction_integer']) - float(json_response['groundtruth']))
            
            #confusion matrix
            y_true.append(int(json_response['groundtruth']))
            y_pred.append(int(json_response['prediction_integer']))
            cm = confusion_matrix(y_true, y_pred)


            evaluated += 1

        except:
            continue
    return {"score": score, "evaluated": evaluated, "accuracy": score/evaluated if evaluated > 0 else 0, "abs_error": abs_error if evaluated > 0 else 0}, cm


def compute_bool_metrics(results):
    """
    Compute boolean metrics from the results of the judge.
    Args:
        results (list): List of results from the judge, each result is a dictionary with keys 'custom_id' and 'response'.
    """
    score = 0
    evaluated = 0
    y_true = []
    y_pred = []
    
    
    for result in results:
        # print(f"Request {result['custom_id']}:")
        # print(f"Response: {result['response']}")
        response = result['response']['body']['choices']['message']['content']

        try:
            json_response = re.search(r'{(.|\r?\n)*}', response).group()
            json_response = json.loads(json_response)
        
            score += json_response['score']
            
            #confusion matrix
            y_true.append(bool(json_response['groundtruth']))
            y_pred.append(bool(json_response['pred']))


            evaluated += 1

        except:
            print("failed extracting json")
            continue
    cm = confusion_matrix(y_true, y_pred)

    return {"score": score, "evaluated": evaluated, "accuracy": score/evaluated if evaluated > 0 else 0}, cm