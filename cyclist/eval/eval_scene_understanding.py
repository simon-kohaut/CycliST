#std lib
from utils.load_env_vars import load_env
load_env()
import json 
import re
import os
from pathlib import Path
import argparse


#external libs
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

#Cyclist
from data.model_wrapper import load_model, BaseModelWrapper, BaseAPIWrapper, GeminiAPIWrapper
from data.dataloader import CyListSceneUnderstandingDataset

parser = argparse.ArgumentParser()

parser.add_argument('--SAMPLED_FRAMES_PER_SEC', type=int, required=True,
    help="Number of frames sampled per second from the video.", default=8)

parser.add_argument('--model_id', type=str, required=True,
    help="Model ID to be used for evaluation, e.g., 'lmms-lab/LLaVA-Video-7B-Qwen2'.", default="lmms-lab/LLaVA-Video-7B-Qwen2")

parser.add_argument('--verbose', action='store_true',
    help="Enable verbose mode to print dataset details.", default=False)

parser.add_argument('--data_path', type=str, required=True,
    help="Path to the video directory.", default="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/LLaVA-cake/Cyclist/data/output")

parser.add_argument('--scene_path', type=str, required=True,
    help="Path to the scene file directory.", default="/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/LLaVA-cake/Cyclist/data/output")


parser.add_argument('--captions_path', default='/pfss/mlde/workspaces/mlde_wsp_Multimodal_on_42/CycliST/Cyclist/output/eval/unicycle_count_answer.csv',
    help="JSON file containing all answers.")

parser.add_argument('--experiment_name', type=str, default="train")




def eval_vqa(args):
    # device ='cuda:2'
        

    # model_id = "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf"
    #model_id = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
    # model_id = "lmms-lab/LLaVA-Video-7B-Qwen2"

    model_wrapper = load_model(args.model_id)


    #output path
    model_id_without_slash= args.model_id.replace("/", "_")
    output_path = Path(args.captions_path , model_id_without_slash)
    
    # create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # answer file
    captions_file_path = Path(output_path, args.experiment_name+".csv")

    #check if we load the data or use the file path 
    return_file_path = False
    if isinstance(model_wrapper, GeminiAPIWrapper):
        return_file_path = True

    # For 16 fps we can use a bs of 4 or 2
    # For 8 fps we can use a bs of 8 
    cyc_dataset = CyListSceneUnderstandingDataset(args.data_path,  args.scene_path,  FPS=32, SAMPLED_FRAMES_PER_SEC=args.SAMPLED_FRAMES_PER_SEC, return_file_path=return_file_path)
    cyc_dataloader = DataLoader(cyc_dataset, batch_size=1, shuffle=True, num_workers=0)#8


    # print one example of the dataset    
    if args.verbose and not return_file_path:
        print("DATASET: ", cyc_dataset)
        for i, data in enumerate(cyc_dataloader):
            print("video shape",data[0].shape)
            print("vid time",data[1])
            print("video frame times",data[2])
            print("scene idx", data[3])
            break
        print("SIZE OF DATASET: ", len(cyc_dataset))


    ### EVAL LOOP
    # save intermediate results to path
    for i, data in enumerate(tqdm(cyc_dataloader)):
        videos = data[0] #[BS, T, H, W, C] -> [[2, 42, 1080, 1920, 3]) 
        video_time = data[1]
        frame_times = data[2]
        scene_idxs = data[3] #scene idx
        
        queries = ["Describe what is shown in the video?"]*len(videos) 

        if isinstance(model_wrapper, BaseModelWrapper):
        #how do i correctly use isinstance
        
            # processor accepts tensors with channel [T, H, W,C] -> (20, 240, 320, 3)
            inputs = model_wrapper.prepare_inputs(videos, queries, video_time, frame_times)


            generate_kwargs = model_wrapper.get_generate_kwargs()
            
            with torch.no_grad():
                output = model_wrapper.model.generate(**inputs, **generate_kwargs)
                generated_text = model_wrapper.decode_outputs(output, inputs)
        else:
            print("Using API wrapper")
            generated_text = model_wrapper.get_outputs(videos, queries, video_time, frame_times)
            
        
        assert len(generated_text) == len(queries), "Generated text and queries must have the same length."

        #save scene understanding answers to csv
        for query, text, scene_idx in zip(queries, generated_text, scene_idxs):

            #remove line breaks from text
            text = text.replace("\n", " ").replace("\"", "\"\"")
            
            # save query, text, answer pairs to a csv file
            with open(captions_file_path, "a") as f:
                f.write(f"\"{query}\";\"{text}\";\"{scene_idx}\"\n")
            
        
# main file 
if __name__ == '__main__':
    args = parser.parse_args()
    eval_vqa(args)