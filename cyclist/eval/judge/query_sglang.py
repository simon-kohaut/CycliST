import pandas as pd
import numpy as np
from openai import OpenAI
import time
import json
import os


def query_judge_llama3(pred, gt, max_tokens=700, judge_prompt=""):
    """
    Receives a prediction and groundtruth as as string and queries the judge model using the OpenAI API.
    """
    
    client = OpenAI(base_url=f"http://127.0.0.1:30000/v1", api_key="None")

    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant helping me score the predictions of a model."},
            {"role": "user",
            "content": judge_prompt.format(pred, gt)},
        ],
        temperature=0,
        max_tokens=700,
    )
    return response.choices[0].message.content

#TODO maybe needed for GPT4 answers
# def query_sglang_llama3_batched(preds, gts):
#     responses = client.chat.completions.create(
#         model="default",
#         messages=[
#             [
#                 {"role": "user", "content": judge_prompt.format(pred, gt)}
#             ]
#             for pred, gt in zip(preds, gts)
#         ]
#     )

#     return [choice.message.content for choice in responses.choices]




def query_judge_batched(answer_csv, judge_prompt):
    """
    Receives a csv file with predictions and groundtruths and queries the judge model in batch using the OpenAI API.
    """

    last_idx =0
    results_all_chunks = []


    client = OpenAI(base_url=f"http://127.0.0.1:30000/v1", api_key="None")


    print("Evaluating", answer_csv)
    # load csv
    df = pd.read_csv(answer_csv, sep=";", names=['query', 'answer','pred','query_type'])
    #display(df.head())

    # types = {"FP": 0, "TP": 0, "FN": 0, "TN": 0}
    # score = 0
    # evaluated = 0

    
    #create batch json to send to llm
    batch_size = len(df)
    for chunk_idx, chunk in enumerate(np.array_split(df, int(len(df)/batch_size))):

        pred = chunk['pred']
        answer = chunk['answer']
        
        requests = []
        for idx, (p, a) in enumerate(zip(pred, answer)):
            requests.append(
                {
                    "custom_id": "request-{}".format(idx+ last_idx),
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
                        "messages": [
                            {"role": "system", "content": "You are a helpful AI assistant helping me score the predictions of a model."},
                            {"role": "user",
                            "content": judge_prompt.format(a, p)},
                        ],
                        "max_tokens": 200,
                    },
                }
            )

        input_file_path = "batch_requests/batch_requests_{}.jsonl".format(chunk_idx)
        # make sure the path exists
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        
        # Write the requests to a file
        with open(input_file_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
                
        # Send the file to the judge via the openAI api
        with open(input_file_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
            
        batch_response = client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        )

        print(f"Batch job created with ID: {batch_response.id}")
        
        # check if batch was completed
        while batch_response.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)
            print(f"Batch job status: {batch_response.status}...trying again in 5 seconds...")
            batch_response = client.batches.retrieve(batch_response.id)

        # if batch was completed, get the results
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

            # for result in results:
            #     print(f"Request {result['custom_id']}:")
            #     # print(f"Response: {result['response']}")
            #     print(result['response']['body']['choices']['message']['content'])

            print("Cleaning up files...")
            # Only delete the result file ID since file_response is just content
            client.files.delete(result_file_id)
        else:
            print(f"Batch job failed with status: {batch_response.status}")
            if hasattr(batch_response, "errors"):
                print(f"Errors: {batch_response.errors}")
        last_idx += idx
        
    # TODO what is in requests
    return results_all_chunks, requests
    