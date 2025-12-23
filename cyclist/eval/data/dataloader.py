from torch.utils.data import Dataset
import os
import json
from eval.utils.load_vid import read_video_pyav
from pathlib import Path

# create custom torch dataset
class CyListVQADataset(Dataset):
    def __init__(self, data_path, question_file = "cycliST_questions.json", FPS=32, SAMPLED_FRAMES_PER_SEC=8, return_file_path=False):
        
        # usually we have the following structure
        # output/
        #     scenes/ #contains single scene files
        #     videos/ #contains all video files
        # cycliST_questions.json #contains all question files in a single file
        # cycliST_scenes.json #contains all scene files in a single file

        # to load a question and an image we use the data stored in the questions file
        # then we load the video which is written in the respective question

        self.data_path = data_path
        self.FRAMES_PER_SEC = FPS
        self.SAMPLED_FRAMES_PER_SEC = SAMPLED_FRAMES_PER_SEC
        self.return_file_path = return_file_path
        
        
        # load questions
        with open(question_file, 'r') as f:
            self.questions = json.load(f)['questions']
        

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # load video
        q = self.questions[idx]
        video_path = Path(self.data_path, self.questions[idx]['video_filename'])

        #for api based model like gemini we just provide the file path
        if self.return_file_path:
            return str(video_path), q['question'], str(q['answer']), [], []
        video_clip, video_time, frame_times = read_video_pyav(video_path, self.FRAMES_PER_SEC, self.SAMPLED_FRAMES_PER_SEC)

        if video_clip.shape[0] != 80:
            print(f"Warning: Video {video_path} has {video_clip.shape[0]} frames, expected 80 frames. Skipping this video.")
        return video_clip, q['question'], str(q['answer']), q['question_family_index'], video_time, frame_times
    




# create custom torch dataset
class CyListSceneUnderstandingDataset(Dataset):
    def __init__(self, data_path, scene_file,  FPS=32, SAMPLED_FRAMES_PER_SEC=8, return_file_path=False):
        
        self.data_path = data_path
        self.FRAMES_PER_SEC = FPS
        self.SAMPLED_FRAMES_PER_SEC = SAMPLED_FRAMES_PER_SEC
        self.return_file_path = return_file_path
        self.scene_file = scene_file
        print("loading from scene file: ", self.scene_file)

        # get a list of all scene files in the directory
        
        
        with open(self.scene_file, 'r') as f:
            self.scenes = json.load(f)['scenes']
                 
        self.scene_dict = {}   
        # keep only a mapping of the scene index to the video
        for idx, scene in enumerate(self.scenes):
            
            self.scene_dict[idx] = (scene['video_file'], scene['scene_index'])
        
        # # drop unnecessary properties
        # for scene in self.scenes:
        #     for obj in scene['objects']:
        #         obj.pop('linear', None)
        #         obj.pop('3d_coords', None)
        #         obj.pop('rotation', None)
        #         obj.pop('pixel_coords', None)
        #         obj.pop('coords_seq', None)
                
        #         #properties which may be useful
        #         obj.pop('enlarge_period', None)
        #         obj.pop('orbit_period', None)
        #         obj.pop('color_period', None)
        #         obj.pop('linear_period', None)
        #         obj.pop('rotate_period', None)

        #         #properties to decode differently
        #         #rename the property
        #         category_name= 'transformations'
        #         obj['transformation'] = {}
        #         obj['transformation']['enlarges'] = obj.pop('enlarge')
        #         obj['transformation']['rotates'] = obj.pop('rotate')
        #         obj['transformation']['orbits'] = obj.pop('orbit')
        #         obj['transformation']['change_colors'] = obj.pop('change_color')
        

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # load video
        file_path, scene_idx = self.scene_dict[idx]
        video_path = Path(self.data_path,file_path)
        
        #for api based model like gemini we just provide the file path
        if self.return_file_path:
            return str(video_path), [], [], scene_idx
        video_clip, video_time, frame_times = read_video_pyav(video_path, self.FRAMES_PER_SEC, self.SAMPLED_FRAMES_PER_SEC)


        if video_clip.shape[0] != 80:
            print(f"Warning: Video {video_path} has {video_clip.shape[0]} frames, expected 80 frames. Skipping this video.")
        return video_clip, video_time, frame_times, scene_idx  #TODO add video idea to get scene information later one

