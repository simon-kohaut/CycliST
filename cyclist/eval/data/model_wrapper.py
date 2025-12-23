#std lib
import requests
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import copy
from abc import ABC, abstractmethod
import time


#external libs
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoTokenizer, AutoModel
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import google.genai as genai
from google.genai import types


# cyclist imports
from eval.utils.conversation import get_conv_template
from eval.utils.intern_video_utils import load_video

def load_model(model_id, device='cuda'):
    if "onevision" in model_id:
        return OneVisionModelWrapper(model_id, device)
    elif 'LLaVA-Video' in model_id:
        return LlavaVideoModelWrapper(model_id, device)
    elif "Intern" in model_id:
        return InternModelWrapper(model_id, device)
    elif "gemini-2.0-flash" in model_id:
        return GeminiAPIWrapper(model_id)
    else:
        raise ValueError("Model id not recognized")


class BaseModelWrapper(ABC):
    def __init__(self, model_id, device='cuda'):
        self.model_id = model_id
        self.device = device
        print(f"Initializing model wrapper for {model_id} on device {device}")

    @abstractmethod
    def prepare_inputs(self, videos, queries, video_time, frame_times):
        pass

    @abstractmethod
    def decode_outputs(self, outputs, inputs):
        pass
    
    @abstractmethod
    def get_generate_kwargs(self):
        """
        Returns the generation configuration for the model.
        This is used to set parameters like max_length, temperature, etc.
        """
        pass
    

    

class OneVisionModelWrapper(BaseModelWrapper):
    def __init__(self, model_id, device='cuda'):
        super().__init__(model_id, device)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map='auto'
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

    def prepare_inputs(self, videos, queries, video_time, frame_times):
        
        #create prompts for each query
        prompts = []
        for query in queries:
            # Each "content" is a list of dicts and you can add image/video/text modalities
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "video"},
                        ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            prompts.append(prompt)

        device_type = next(iter(self.model.parameters())).device.type
        inputs = self.processor(text=prompts, videos=torch.unbind(videos, dim=0), padding=True, return_tensors="pt").to(device_type, torch.float16)
        return inputs
    
    def decode_outputs(self, outputs, inputs):
         # Find the eos token id in the inputs
        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_token_pos = torch.where(inputs["input_ids"] == eos_token_id)[1]

        # Extract only the newly generated tokens
        new_tokens = []
        for i, eos_pos in enumerate(eos_token_pos):
            new_tokens.append(outputs[i, eos_pos + 1:])

        # Decode the newly generated tokens
        generated_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return generated_text
    
    def get_generate_kwargs(self):
        return {"max_new_tokens": 300, "do_sample": False, "top_p": 0.9}

            

class LlavaVideoModelWrapper(BaseModelWrapper):
    def __init__(self, model_id, device='cuda'):
        super().__init__(model_id, device)
        model_name = "llava_qwen"
        device_map = "auto"
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_id, None, model_name, torch_dtype="bfloat16", device_map=device_map
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.max_length = max_length

    def prepare_inputs(self, videos, queries, video_time, frame_times):
        #process videos
        videos = [self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16() for video in videos]

        #create prompts
        prompts = []
        for vidx,video in enumerate(videos):
            # BUILD PROMPT
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            time_instruciton = f"The video lasts for {video_time[vidx]:.2f} seconds, and {len(video)} frames are uniformly sampled from it. These frames are located at {list(torch.stack(frame_times)[:,vidx].numpy())}.Please answer the following question related to this video."
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n"+queries[vidx]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            #print(prompt_question)
            prompts.append(prompt_question)

        # create input_ids from prompts by tokenizing them
        device_type = next(iter(self.model.parameters())).device.type
        input_ids = [tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device_type) for p in prompts]

        # get maximum shape of input_ids
        max_shape = max([i.shape[1] for i in input_ids])

        input_ids_padded = torch.ones(len(input_ids), max_shape, dtype=torch.long) * self.tokenizer.pad_token_id
        
        #add input_ids to padded tensor
        for i, input_id in enumerate(input_ids):
            input_ids_padded[i, max_shape-input_id.shape[1]:] = input_id

        inputs = {
        "inputs": input_ids_padded,
        "images": videos,
        "modalities": ["video"]*len(videos)
        }
        return inputs

    def decode_outputs(self, outputs, inputs):
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    def get_generate_kwargs(self):
        return {"do_sample": True, "temperature": 0.9, "max_new_tokens": 4096}


class InternModelWrapper(BaseModelWrapper):
    def __init__(self, model_id, device='cuda'):
        super().__init__(model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).half().cuda().to(torch.bfloat16)
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.num_segments = 128
        

    def prepare_inputs2(self,  videos, queries, video_time, frame_times):


        for video, query in zip(videos, queries):
            #obtain pixel values and number of patches
            pixel_values, num_patches_list = load_video(video, num_segments=self.num_segments, max_num=1, get_frame_by_duration=False)
            
            
            
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            
            # set up question
            question = video_prefix + query
            
            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

            # get img context token id
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
            self.model.img_context_token_id = img_context_token_id

            # get template and eos token id
            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

            #apply the template to the question
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            # replace <image> token with the image tokens
            for num_patches in num_patches_list:
                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + self.IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            # tokenize the query
            model_inputs = self.tokenizer(query, return_tensors='pt')
            device = torch.device(self.model.language_model.device if torch.cuda.is_available() else 'cpu')
            input_ids = model_inputs['input_ids'].to(device)
            attention_mask = model_inputs['attention_mask'].to(device)

            return {
                "pixel_values": pixel_values,                
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def prepare_inputs(self,  videos, queries, video_time, frame_times):

        pixel_values =[]
        num_patches_list = []
        for video in videos:
            # load video and get pixel values
            pixel_value, num_patches = load_video(video, num_segments=self.num_segments, max_num=1, get_frame_by_duration=False)
            pixel_values.append(pixel_value.to(torch.float16).cuda())
            num_patches_list.append(num_patches)
        
        pixel_values = torch.cat(pixel_values, dim=0)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        
        
        self.template = get_conv_template(self.model.template)
        self.template.system_message = self.model.system_message
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.template.sep.strip())

        prompts = []
        for idx, num_patches in enumerate(num_patches_list):
            # set up question
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list[idx]))])
            
            question = video_prefix + queries[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            self.template = get_conv_template(self.model.template)
            self.template.system_message = self.model.system_message
            
            # get template and eos token id
            self.template.append_message(self.template.roles[0], question)
            self.template.append_message(self.template.roles[1], None)
            prompt = self.template.get_prompt()

            # # replace <image> token with the image tokens
            # image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + self.IMG_END_TOKEN
            # prompt = prompt.replace('<image>', image_tokens, 1)
                        # replace <image> token with the image tokens
            for np in num_patches:
                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * np + self.IMG_END_TOKEN
                prompt = prompt.replace('<image>', image_tokens, 1)
            prompts.append(prompt)

        
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
            
        return {
                "pixel_values": pixel_values,                
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
       

    def decode_outputs(self, outputs, inputs):
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return response
    
    def get_generate_kwargs(self):
        return {"do_sample":False, "temperature":0.0, "max_new_tokens":1024, "top_p":0.1, "num_beams":1, "eos_token_id": self.eos_token_id}
        
        
    

class BaseAPIWrapper:
    def __init__(self, model_id):
        self.model_id = model_id
        print(f"Initialized BaseAPIWrapper for {model_id}")

    def get_outputs(self, videos, queries, video_time, frame_times):
       pass



    
class GeminiAPIWrapper(BaseAPIWrapper):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.client = genai.Client()
        self.model_id = model_id
        
        self.system_instruction= "When given a video and a query, call the relevant function only once with the appropriate timecodes and text for the video"
        
        print(f"Initialized GeminiAPIWrapper for {model_id}")

    def get_outputs(self, videos, queries, video_time, frame_times):
        
        uploaded_videos = [self._upload_video(video) for video in videos]
        responses = []
        for uploaded_video, query in zip(uploaded_videos, queries):
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    uploaded_video,
                    query,
                ]
            )
            responses.append(response.text)
        return responses
        
    
    def _upload_video(self, video_file_name):
        video_file = self.client.files.upload(file=video_file_name)

        while video_file.state == "PROCESSING":
            print('Waiting for video to be processed.')
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(video_file.state)
        print(f'Video processing complete: ' + video_file.uri)

        return video_file







