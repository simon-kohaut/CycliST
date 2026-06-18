from pathlib import Path
import argparse

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from cyclist.eval.utils.load_env_vars import load_env
load_env()

from cyclist.eval.data.model_wrapper import load_model, BaseModelWrapper, GeminiAPIWrapper
from cyclist.eval.data.dataloader import CyListVQADataset

parser = argparse.ArgumentParser()
parser.add_argument('--SAMPLED_FRAMES_PER_SEC', type=int, required=True,
    help="Number of frames sampled per second from the video.", default=8)
parser.add_argument('--model_id', type=str, required=True,
    help="Model ID, e.g. 'lmms-lab/LLaVA-Video-7B-Qwen2'.")
parser.add_argument('--data_path', type=str, required=True,
    help="Path to the video directory.")
parser.add_argument('--answer_path', type=str, required=True,
    help="Path to save the generated answers.")
parser.add_argument('--question_file', type=str, required=True,
    help="JSON file containing all questions.")
parser.add_argument('--experiment_name', type=str, default="train")
parser.add_argument('--verbose', action='store_true', default=False)


def eval_vqa(args):
    model_wrapper = load_model(args.model_id)

    # output is written to answer_path/{model_id}/{experiment_name}.csv
    answer_path = Path(args.answer_path, args.model_id.replace("/", "_"))
    answer_path.mkdir(parents=True, exist_ok=True)
    answer_file_path = answer_path / (args.experiment_name + ".csv")

    return_file_path = isinstance(model_wrapper, GeminiAPIWrapper)
    cyc_dataset = CyListVQADataset(args.data_path, args.question_file,
                                   FPS=32, SAMPLED_FRAMES_PER_SEC=args.SAMPLED_FRAMES_PER_SEC,
                                   return_file_path=return_file_path)
    batch_size = 2 if args.SAMPLED_FRAMES_PER_SEC > 16 else 4
    cyc_dataloader = DataLoader(cyc_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if args.verbose and not return_file_path:
        for data in cyc_dataloader:
            print("video shape", data[0].shape)
            print("query", data[1])
            print("answer", data[2])
            print("family_index", data[3])
            break
        print("dataset size:", len(cyc_dataset))

    for data in tqdm(cyc_dataloader):
        videos, queries, answers, query_family_index, video_time, frame_times = data

        if isinstance(model_wrapper, BaseModelWrapper):
            inputs = model_wrapper.prepare_inputs(videos, queries, video_time, frame_times)
            with torch.no_grad():
                output = model_wrapper.model.generate(**inputs, **model_wrapper.get_generate_kwargs())
                generated_text = model_wrapper.decode_outputs(output, inputs)
        else:
            generated_text = model_wrapper.get_outputs(videos, queries, video_time, frame_times)

        assert len(generated_text) == len(queries)

        with open(answer_file_path, "a") as f:
            for query, text, answer, family_idx in zip(queries, generated_text, answers, query_family_index):
                text = text.replace("\n", " ").replace("\"", "\"\"")
                f.write(f"\"{query}\";\"{text}\";\"{answer}\";\"{family_idx}\"\n")


if __name__ == '__main__':
    eval_vqa(parser.parse_args())
