import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json
from pathlib import Path


def read_and_update_offset(add_to_offset):
    offset_file_path = "offset.json"
    with open(offset_file_path, 'r') as f:
        offset_data = json.load(f)
    offset = offset_data['offset']
    with open(offset_file_path, 'w') as f:
        json.dump({'offset': offset + add_to_offset}, f)
    return offset + add_to_offset


def query_judge_llama3(pred, gt, max_tokens=700, judge_prompt=""):
    """
    Receives a prediction and groundtruth as a string and queries the judge model using the OpenAI API.
    """
    client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant helping me score the predictions of a model."},
            {"role": "user", "content": judge_prompt.format(pred, gt)},
        ],
        temperature=0,
        max_tokens=700,
    )
    return response.choices[0].message.content


def query_judge_batched(answer_csv, judge_prompt):
    """
    Reads a CSV of predictions and queries the SGLang judge concurrently via chat completions.
    All rows are submitted in parallel (matching the throughput of the old batch API).
    """
    client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")

    print("Evaluating", answer_csv)
    df = pd.read_csv(answer_csv, sep=";", names=['query', 'pred', 'gt_answer', 'family_index'])
    df['pred'] = df['pred'].apply(lambda x: x if len(x) <= 10000 else x[:10000])

    offset = read_and_update_offset(len(df))
    base_idx = offset - len(df)

    rows = [(i, str(r['pred']), str(r['gt_answer'])) for i, (_, r) in enumerate(df.iterrows())]

    def call_one(i, p, gt):
        msgs = [
            {"role": "system", "content": "You are a helpful AI assistant helping me map the predictions of a model to a json."},
            {"role": "user", "content": judge_prompt.format(p, gt)},
        ]
        resp = client.chat.completions.create(model="default", messages=msgs, temperature=0, max_tokens=200)
        return i, resp.choices[0].message.content, msgs

    results = [None] * len(rows)
    requests = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(call_one, i, p, gt): i for i, p, gt in rows}
        for future in as_completed(futures):
            i, content, msgs = future.result()
            custom_id = f"request-{base_idx + i}"
            results[i] = {"custom_id": custom_id, "response": {"body": {"choices": [{"message": {"content": content}}]}}}
            requests[i] = {"custom_id": custom_id, "method": "POST", "url": "/chat/completions",
                           "body": {"model": "meta-llama/Meta-Llama-3-70B-Instruct", "messages": msgs, "max_tokens": 200}}

    return results, requests


def prepare_scene_config(file_name, scenes_base_path="output/scenes"):
    if "unicycle" in file_name:
        path = Path(scenes_base_path, "unicycle", "test")
    elif "bicycle" in file_name:
        path = Path(scenes_base_path, "bicycle", "test")
    elif "tricycle" in file_name:
        path = Path(scenes_base_path, "tricycle", "test")
    else:
        raise ValueError(f"Cannot determine dataset from file name: {file_name}")

    with open(Path(path, file_name), 'r') as f:
        scene_config_json = json.load(f)
        objects = scene_config_json['objects']

        for obj in objects:
            obj.pop('angle')
            obj.pop('location')
            obj.pop('always_within_boundaries')

            if 'intermittent_location' in obj:
                obj.pop('intermittent_location')

            for cycle_type in ['linear', 'orbit', 'recolor', 'resize', 'rotate']:
                if 'cycles' in obj and cycle_type in obj['cycles']:
                    obj['cycles'].pop(cycle_type)
                    obj[cycle_type] = "True"
                else:
                    obj[cycle_type] = "not_mentioned"

            if 'cycles' in obj:
                obj.pop('cycles')

    return objects


def query_judge_su_batched(answer_csv, judge_prompt):
    """
    Reads a CSV of scene-understanding predictions and queries the SGLang judge concurrently.
    All rows are submitted in parallel (matching the throughput of the old batch API).
    Returns results, requests, gt_objects, and predictions.
    """
    client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")

    print("Evaluating", answer_csv)
    df = pd.read_csv(answer_csv, sep=";", names=['query', 'pred', 'scene_config'])

    offset = read_and_update_offset(len(df))
    base_idx = offset - len(df)

    rows = [(i, str(r['pred'])) for i, (_, r) in enumerate(df.iterrows())]

    def call_one(i, p):
        msgs = [
            {"role": "system", "content": "You are a helpful AI assistant helping me map the predictions of a model to a json."},
            {"role": "user", "content": judge_prompt.format(p)},
        ]
        resp = client.chat.completions.create(model="default", messages=msgs, temperature=0, max_tokens=4000)
        return i, resp.choices[0].message.content, msgs

    results = [None] * len(rows)
    requests = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(call_one, i, p): i for i, p in rows}
        for future in as_completed(futures):
            i, content, msgs = future.result()
            custom_id = f"request-{base_idx + i}"
            results[i] = {"custom_id": custom_id, "response": {"body": {"choices": [{"message": {"content": content}}]}}}
            requests[i] = {"custom_id": custom_id, "method": "POST", "url": "/chat/completions",
                           "body": {"model": "meta-llama/Meta-Llama-3-70B-Instruct", "messages": msgs, "max_tokens": 4000}}

    objects_list = [prepare_scene_config(scene_config) for scene_config in df['scene_config'].values]
    pred = df['pred']

    return results, requests, objects_list, pred
