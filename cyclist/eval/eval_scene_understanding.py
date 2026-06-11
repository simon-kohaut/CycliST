from pathlib import Path
import argparse

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from utils.load_env_vars import load_env
load_env()

from data.model_wrapper import load_model, BaseModelWrapper, GeminiAPIWrapper
from data.dataloader import CyListSceneUnderstandingDataset

parser = argparse.ArgumentParser()
parser.add_argument('--SAMPLED_FRAMES_PER_SEC', type=int, required=True,
    help="Number of frames sampled per second from the video.", default=8)
parser.add_argument('--model_id', type=str, required=True,
    help="Model ID, e.g. 'lmms-lab/LLaVA-Video-7B-Qwen2'.")
parser.add_argument('--data_path', type=str, required=True,
    help="Path to the video directory.")
parser.add_argument('--scene_path', type=str, required=True,
    help="Path to the scene JSON file.")
parser.add_argument('--captions_path', type=str, required=True,
    help="Path to save the generated captions.")
parser.add_argument('--experiment_name', type=str, default="train")
parser.add_argument('--verbose', action='store_true', default=False)


def eval_scene_understanding(args):
    model_wrapper = load_model(args.model_id)

    # output is written to captions_path/{model_id}/{experiment_name}.csv
    output_path = Path(args.captions_path, args.model_id.replace("/", "_"))
    output_path.mkdir(parents=True, exist_ok=True)
    captions_file_path = output_path / (args.experiment_name + ".csv")

    return_file_path = isinstance(model_wrapper, GeminiAPIWrapper)
    cyc_dataset = CyListSceneUnderstandingDataset(args.data_path, args.scene_path,
                                                  FPS=32, SAMPLED_FRAMES_PER_SEC=args.SAMPLED_FRAMES_PER_SEC,
                                                  return_file_path=return_file_path)
    cyc_dataloader = DataLoader(cyc_dataset, batch_size=1, shuffle=True, num_workers=0)

    if args.verbose and not return_file_path:
        for data in cyc_dataloader:
            print("video shape", data[0].shape)
            print("vid time", data[1])
            print("frame times", data[2])
            print("scene idx", data[3])
            break
        print("dataset size:", len(cyc_dataset))

    for data in tqdm(cyc_dataloader):
        videos, video_time, frame_times, scene_idxs = data
        queries = ["Describe what is shown in the video?"] * len(videos)

        if isinstance(model_wrapper, BaseModelWrapper):
            inputs = model_wrapper.prepare_inputs(videos, queries, video_time, frame_times)
            with torch.no_grad():
                output = model_wrapper.model.generate(**inputs, **model_wrapper.get_generate_kwargs())
                generated_text = model_wrapper.decode_outputs(output, inputs)
        else:
            generated_text = model_wrapper.get_outputs(videos, queries, video_time, frame_times)

        assert len(generated_text) == len(queries)

        with open(captions_file_path, "a") as f:
            for query, text, scene_idx in zip(queries, generated_text, scene_idxs):
                text = text.replace("\n", " ").replace("\"", "\"\"")
                f.write(f"\"{query}\";\"{text}\";\"{scene_idx}\"\n")


if __name__ == '__main__':
    eval_scene_understanding(parser.parse_args())
