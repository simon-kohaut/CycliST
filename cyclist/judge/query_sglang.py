import pandas as pd
import numpy as np
from openai import OpenAI
import time
import json
import os
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
    Receives a csv file with predictions and groundtruths and queries the judge model in batch using the OpenAI API.
    """
    last_idx = 0
    results_all_chunks = []

    client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")

    print("Evaluating", answer_csv)
    df = pd.read_csv(answer_csv, sep=";", names=['query', 'pred', 'gt_answer', 'family_index'])
    df['pred'] = df['pred'].apply(lambda x: x if len(x) <= 10000 else x[:10000])

    offset = read_and_update_offset(len(df))

    batch_size = len(df)
    for chunk_idx, chunk in enumerate(np.array_split(df, int(len(df)/batch_size))):

        pred = chunk['pred']
        gt_answer = chunk['gt_answer']

        requests = []
        for idx, (p, gt) in enumerate(zip(pred, gt_answer)):
            requests.append(
                {
                    "custom_id": "request-{}".format(offset + idx),
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
                        "messages": [
                            {"role": "system", "content": "You are a helpful AI assistant helping me map the predictions of a model to a json."},
                            {"role": "user", "content": judge_prompt.format(p, gt)},
                        ],
                        "max_tokens": 200,
                    },
                }
            )

        input_file_path = "batch_requests/batch_requests_{}_{}.jsonl".format(chunk_idx, np.random.randint(100000))
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)

        with open(input_file_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        with open(input_file_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")

        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print("FILE response ID", file_response.id)
        print(f"Batch job created with ID: {batch_response.id}")

        while batch_response.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)
            print(f"Batch job status: {batch_response.status}...trying again in 5 seconds...")
            batch_response = client.batches.retrieve(batch_response.id)

        if batch_response.status == "completed":
            print("Batch job completed successfully!")
            print(f"Request counts: {batch_response.request_counts}")

            print("loading data from file", batch_response.output_file_id)
            result_file_id = batch_response.output_file_id
            file_response = client.files.content(result_file_id)
            result_content = file_response.read().decode("utf-8")

            results = [
                json.loads(line) for line in result_content.split("\n") if line.strip() != ""
            ]
            results_all_chunks.extend(results)

            print("Cleaning up files...")
            client.files.delete(result_file_id)
        else:
            print(f"Batch job failed with status: {batch_response.status}")
            if hasattr(batch_response, "errors"):
                print(f"Errors: {batch_response.errors}")

        last_idx += idx

    return results_all_chunks, requests


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
    Receives a csv file with predictions and queries the judge model in batch using the OpenAI API.
    Returns results, requests, gt_objects, and predictions.
    """
    last_idx = 0
    results_all_chunks = []

    client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="None")

    print("Evaluating", answer_csv)
    df = pd.read_csv(answer_csv, sep=";", names=['query', 'pred', 'scene_config'])

    offset = read_and_update_offset(len(df))

    batch_size = len(df)
    for chunk_idx, chunk in enumerate(np.array_split(df, int(len(df)/batch_size))):

        pred = chunk['pred']

        requests = []
        for idx, p in enumerate(pred):
            requests.append(
                {
                    "custom_id": "request-{}".format(offset + idx),
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
                        "messages": [
                            {"role": "system", "content": "You are a helpful AI assistant helping me map the predictions of a model to a json."},
                            {"role": "user", "content": judge_prompt.format(p)},
                        ],
                        "max_tokens": 4000,
                    },
                }
            )

        input_file_path = "batch_requests/batch_requests_{}_{}.jsonl".format(chunk_idx, np.random.randint(100000))
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)

        with open(input_file_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        with open(input_file_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")

        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print("FILE response ID", file_response.id)
        print(f"Batch job created with ID: {batch_response.id}")

        while batch_response.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)
            print(f"Batch job status: {batch_response.status}...trying again in 5 seconds...")
            batch_response = client.batches.retrieve(batch_response.id)

        if batch_response.status == "completed":
            print("Batch job completed successfully!")
            print(f"Request counts: {batch_response.request_counts}")

            print("loading data from file", batch_response.output_file_id)
            result_file_id = batch_response.output_file_id
            file_response = client.files.content(result_file_id)
            result_content = file_response.read().decode("utf-8")

            results = [
                json.loads(line) for line in result_content.split("\n") if line.strip() != ""
            ]
            results_all_chunks.extend(results)

            print("Cleaning up files...")
            client.files.delete(result_file_id)
        else:
            print(f"Batch job failed with status: {batch_response.status}")
            if hasattr(batch_response, "errors"):
                print(f"Errors: {batch_response.errors}")

        last_idx += idx

    objects_list = [prepare_scene_config(scene_config) for scene_config in df['scene_config'].values]

    return results_all_chunks, requests, objects_list, pred
