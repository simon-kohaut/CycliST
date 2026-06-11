import json
from pathlib import Path
from torch.utils.data import Dataset
from eval.data.load_vid import read_video_pyav


class CyListVQADataset(Dataset):
    def __init__(self, data_path, question_file="cycliST_questions.json", FPS=32, SAMPLED_FRAMES_PER_SEC=8, return_file_path=False):
        self.data_path = data_path
        self.FRAMES_PER_SEC = FPS
        self.SAMPLED_FRAMES_PER_SEC = SAMPLED_FRAMES_PER_SEC
        self.return_file_path = return_file_path
        self.expected_frames = int(160 * SAMPLED_FRAMES_PER_SEC / FPS)

        with open(question_file, 'r') as f:
            self.questions = json.load(f)['questions']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        video_path = Path(self.data_path, q['video_filename'])

        if self.return_file_path:
            return str(video_path), q['question'], str(q['answer']), [], []

        video_clip, video_time, frame_times = read_video_pyav(video_path, self.FRAMES_PER_SEC, self.SAMPLED_FRAMES_PER_SEC)

        if video_clip.shape[0] != self.expected_frames:
            print(f"Warning: {video_path} has {video_clip.shape[0]} frames, expected {self.expected_frames}.")
        return video_clip, q['question'], str(q['answer']), q['question_family_index'], video_time, frame_times


class CyListSceneUnderstandingDataset(Dataset):
    def __init__(self, data_path, scene_file, FPS=32, SAMPLED_FRAMES_PER_SEC=8, return_file_path=False):
        self.data_path = data_path
        self.FRAMES_PER_SEC = FPS
        self.SAMPLED_FRAMES_PER_SEC = SAMPLED_FRAMES_PER_SEC
        self.return_file_path = return_file_path
        self.expected_frames = int(160 * SAMPLED_FRAMES_PER_SEC / FPS)

        with open(scene_file, 'r') as f:
            self.scenes = json.load(f)['scenes']

        self.scene_dict = {idx: (scene['video_file'], scene['scene_index']) for idx, scene in enumerate(self.scenes)}

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        file_path, scene_idx = self.scene_dict[idx]
        video_path = Path(self.data_path, file_path)

        if self.return_file_path:
            return str(video_path), [], [], scene_idx

        video_clip, video_time, frame_times = read_video_pyav(video_path, self.FRAMES_PER_SEC, self.SAMPLED_FRAMES_PER_SEC)

        if video_clip.shape[0] != self.expected_frames:
            print(f"Warning: {video_path} has {video_clip.shape[0]} frames, expected {self.expected_frames}.")
        return video_clip, video_time, frame_times, scene_idx
