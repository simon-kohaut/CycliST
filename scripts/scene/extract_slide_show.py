import os
import cv2
import argparse
import numpy as np

def extract_frames_from_videos(path: str, output_dir: str, prefix: str):
    if os.path.exists(path):
        output_frames_dir = output_dir
        if not os.path.isdir(output_frames_dir):
            print(f"Creating output directory: {output_frames_dir}")
        os.makedirs(output_frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Error: Video {path} has no frames.")
            cap.release()
            return

        num_frames_to_extract = 4
        frame_indices_to_extract = np.linspace(0, total_frames, num=num_frames_to_extract, endpoint=False, dtype=int)

        for i, frame_index in enumerate(frame_indices_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_frames_dir, f"{prefix}_{i}.png")
                cv2.imwrite(frame_filename, frame)
            else:
                print(f"Warning: Could not read frame {frame_index} from video {path}.")

        cap.release()
        print(f"Extracted frames from {path} to {output_frames_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video at 1-second intervals.")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="The path to the input video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output/frames",
        help="The directory to save extracted frames.",
    )
    parser.add_argument("--prefix", type=str, default="frame", help="Prefix for the output frame filenames.")
    args = parser.parse_args()
    extract_frames_from_videos(args.video_path, args.output_dir, args.prefix)
