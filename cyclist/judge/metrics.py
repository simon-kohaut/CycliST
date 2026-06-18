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
    hits = 0
    evaluated = 0
    y_true = []
    y_pred = []
    abs_error = 0

    for result in results:
        response = result['response']['body']['choices'][0]['message']['content']

        try:
            regex = r'\{[\s\S]*?\}'
            json_response = re.search(regex, response).group()
            json_response = json.loads(json_response)

            pred = json_response['prediction_integer']
            gt = json_response['groundtruth']

            abs_error += np.abs(float(pred) - float(gt))

            y_true.append(int(json_response['groundtruth']))
            y_pred.append(int(json_response['prediction_integer']))
            cm = confusion_matrix(y_true, y_pred)

            if int(pred) == int(gt):
                hits += 1
            evaluated += 1

        except Exception as e:
            print("failed extracting json")
            print(e)
            continue

    return {"hits": hits, "evaluated": evaluated, "accuracy": hits/evaluated if evaluated > 0 else 0, "avg_abs_error": abs_error/evaluated if evaluated > 0 else 0, "y_true": y_true, "y_pred": y_pred}, cm


def str_to_bool(s: str) -> bool:
    if s == 2 or s == "2":
        return 2

    s = s.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return 1
    elif s in ("false", "0", "no", "n", "f"):
        return 0
    else:
        raise ValueError(f"Cannot convert {s!r} to boolean")


def compute_bool_metrics(results):
    """
    Compute boolean metrics from the results of the judge.
    Args:
        results (list): List of results from the judge, each result is a dictionary with keys 'custom_id' and 'response'.
    """
    evaluated = 0
    undecided = 0
    y_true = []
    y_pred = []
    hits = 0

    for result in results:
        response = result['response']['body']['choices'][0]['message']['content']

        try:
            regex = r'\{[\s\S]*?\}'
            json_response = re.search(regex, response).group()
            json_response = json.loads(json_response)

            gt = str_to_bool(json_response['groundtruth'])
            pred = str_to_bool(json_response['pred'])

            if pred == 2:
                undecided += 1

            y_true.append(gt)
            y_pred.append(pred)

            if gt == pred:
                hits += 1
            evaluated += 1

        except Exception as e:
            print("failed extracting json")
            print(e)
            continue

    cm = confusion_matrix(y_true, y_pred)
    return {"hits": hits, "evaluated": evaluated, "undecided": undecided, "accuracy": hits/evaluated if evaluated > 0 else 0}, cm


def compute_attribute_metrics(results):
    """
    Compute attribute metrics from the results of the judge.
    Args:
        results (list): List of results from the judge, each result is a dictionary with keys 'custom_id' and 'response'.
    """
    evaluated = 0
    undecided = 0
    y_true = []
    y_pred = []
    hits = 0

    for result in results:
        response = result['response']['body']['choices'][0]['message']['content']

        try:
            regex = r'\{[\s\S]*?\}'
            json_response = re.search(regex, response).group()
            json_response = json.loads(json_response)

            pred = json_response['pred']
            gt = json_response['groundtruth']

            if pred == "undetermined":
                undecided += 1

            y_true.append(gt)
            y_pred.append(pred)

            if gt == pred:
                hits += 1
            evaluated += 1

        except Exception as e:
            print("failed extracting json")
            print(e)
            continue

    cm = confusion_matrix(y_true, y_pred)
    return {"hits": hits, "evaluated": evaluated, "undecided": undecided, "accuracy": hits/evaluated if evaluated > 0 else 0}, cm


def matches(pred, gt):
    if pred['mesh'] != "not_mentioned":
        if pred['mesh'] != gt['mesh']:
            return False
    if pred['material'] != "not_mentioned":
        if pred['material'] != gt['material']:
            return False
    if pred['size'] != "not_mentioned":
        if pred['size'] != gt['size']:
            return False
    if pred['color'] != "not_mentioned":
        if pred['color'] != gt['color']:
            return False

    if (pred['mesh'] == "not_mentioned" and pred['material'] == "not_mentioned"
            and pred['size'] == "not_mentioned" and pred['color'] == "not_mentioned"):
        return False

    return True


def is_true(v):
    if isinstance(v, str):
        return v in ("true", "True")
    return v


def match_and_compute_metric(pred, gt):
    pred_sorted = []

    for obj in pred:
        for key in obj:
            if isinstance(obj[key], str):
                obj[key] = obj[key].lower()
    for obj in gt:
        for key in obj:
            if isinstance(obj[key], str):
                obj[key] = obj[key].lower()

    for i in range(0, 4):
        for pidx in range(0, len(pred)):
            count_not_mentioned = 0
            for attr in ('mesh', 'material', 'size', 'color'):
                if attr in pred[pidx] and pred[pidx][attr] == "not_mentioned":
                    count_not_mentioned += 1
            if count_not_mentioned == i:
                pred_sorted.append(pred[pidx])

    match_indices_list = []
    matched_gt = []
    matched_pred = []

    for pidx in range(0, len(pred_sorted)):
        for gidx in range(0, len(gt)):
            if matches(pred_sorted[pidx], gt[gidx]):
                if gidx not in matched_gt and pidx not in matched_pred:
                    match_indices_list.append((pidx, gidx))
                    matched_gt.append(gidx)
                    matched_pred.append(pidx)

    cycle_match = 0
    cycle_pred = 0
    cycle_gt = 0

    for pidx, gidx in match_indices_list:
        for cycle in ["linear", "orbit", "rotate", "resize", "recolor"]:
            if is_true(gt[gidx][cycle]) and is_true(pred[pidx][cycle]):
                cycle_match += 1
    for ps in pred_sorted:
        for cycle in ["linear", "orbit", "rotate", "resize", "recolor"]:
            if is_true(ps[cycle]):
                cycle_pred += 1
    for g in gt:
        for cycle in ["linear", "orbit", "rotate", "resize", "recolor"]:
            if is_true(g[cycle]):
                cycle_gt += 1

    obj_precision = len(match_indices_list) / len(pred)
    obj_recall = len(match_indices_list) / len(gt)
    cycle_precision = cycle_match / cycle_pred if cycle_pred > 0 else 0
    cycle_recall = cycle_match / cycle_gt if cycle_gt > 0 else 0

    return obj_precision, obj_recall, cycle_precision, cycle_recall


def compute_match_metrics(results, gt_objects, responses):
    obj_precision_list = []
    obj_recall_list = []
    obj_f1_scores = []
    cycle_precision_list = []
    cycle_recall_list = []
    cycle_f1_scores = []
    evaluated = 0

    for idx, (result, gtobjs) in enumerate(zip(results, gt_objects)):
        response = result['response']['body']['choices'][0]['message']['content']

        try:
            regex = r'(\{\s*\"mapped_objects\"\s*:\s*\[.*?\]\s*\})'
            parsed_json_response = re.search(regex, response, re.DOTALL).group()
            json_response = json.loads(parsed_json_response)

            obj_precision, obj_recall, cycle_precision, cycle_recall = match_and_compute_metric(json_response['mapped_objects'], gtobjs)
            obj_precision_list.append(obj_precision)
            obj_recall_list.append(obj_recall)
            obj_f1 = 2 * (obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0
            obj_f1_scores.append(obj_f1)

            cycle_precision_list.append(cycle_precision)
            cycle_recall_list.append(cycle_recall)
            cycle_f1 = 2 * (cycle_precision * cycle_recall) / (cycle_precision + cycle_recall) if (cycle_precision + cycle_recall) > 0 else 0
            cycle_f1_scores.append(cycle_f1)
            evaluated += 1

        except Exception as e:
            print("failed extracting json")
            print(e)
            continue

    obj_precision = sum(obj_precision_list) / evaluated
    obj_recall = sum(obj_recall_list) / evaluated
    obj_f1_score = sum(obj_f1_scores) / evaluated
    cycle_precision = sum(cycle_precision_list) / evaluated
    cycle_recall = sum(cycle_recall_list) / evaluated
    cycle_f1_score = sum(cycle_f1_scores) / evaluated

    return {"evaluated": evaluated, "obj_precision": obj_precision, "obj_recall": obj_recall, "obj_f1_score": obj_f1_score,
            "cycle_precision": cycle_precision, "cycle_recall": cycle_recall, "cycle_f1_score": cycle_f1_score}
