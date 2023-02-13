import os

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np

from opencv_transforms.transforms import (
    Compose, Normalize, Resize, CenterCrop, ToTensor
)

import cv2

# from pytorchvideo.data.encoded_video import EncodedVideo

# from torchvision.transforms import Compose, Lambda, Resize
# from torchvision.transforms._transforms_video import (
#     CenterCropVideo,
#     NormalizeVideo
# )
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     UniformTemporalSubsample
# )

from config import *

class HateSpeechData(data.Dataset):

    def __init__(self, args, phase):
        if args.multilabel:
            self.mapping = {"Organisation": 0, "Location": 1, "Individual": 2, "Community": 3, "None": 4}
        self.args = args
        if phase == 'train':
            self.comments = self.load_comments(self.args.train_question_file)
        elif phase == 'validation':
            self.comments = self.load_comments(self.args.validation_question_file)
        else:
            self.comments = self.load_comments(self.args.test_question_file)

        self.comments[['videoID', 'source']] = self.comments['url'].apply(lambda x: self.parse_urls(x))
        
        if args.add_video:
            # self.transform = ApplyTransformToKey(
            #     key="video",
            #     transform=Compose(
            #         [
            #             UniformTemporalSubsample(NUM_FRAMES),
            #             Resize(RESIZE),
            #             CenterCropVideo(crop_size=CROP_SIZE),
            #             Lambda(lambda x: x / 255.0),
            #             NormalizeVideo(MEAN, STD)
            #         ]
            #     ),
            # )
            self.transform = Compose(
                [
                    Resize(RESIZE),
                    CenterCrop(CROP_SIZE),
                    # Lambda(lambda x: x / 255.0),
                    ToTensor(),
                    Normalize(MEAN, STD)
                ]
            )

        if args.add_other_comments:
            self.other_comments_data = self.load_metadata(self.args.other_comments_path)
            self.comments = pd.merge(self.comments, self.other_comments_data, how='left', on=['url', 'comment'])
        if args.add_title or args.add_description or args.add_transcription:
            self.metadata = self.load_metadata(args.metadata_path)
            self.metadata = self.metadata.replace(np.nan, '', regex=True)
            if "bert" in self.args.model:
                if self.args.desc_keyphrase_extract:
                    self.metadata['desc'] = self.metadata['key_phrases_desc_bert']
                if self.args.transcript_keyphrase_extract:
                    self.metadata['transcript'] = self.metadata['key_phrases_transcript_bert']
            elif "longformer" in self.args.model:
                if self.args.desc_keyphrase_extract:
                    self.metadata['desc'] = self.metadata['key_phrases_desc_long']
                if self.args.transcript_keyphrase_extract:
                    self.metadata['transcript'] = self.metadata['key_phrases_transcript_long']
            self.comments = pd.merge(self.comments, self.metadata, how='left', on='url')
        self.comments = self.comments.replace(np.nan, '', regex=True)
        self.comments.drop_duplicates(inplace=True)
        if self.args.remove_none:
            self.comments = self.comments[self.comments['label'] == True]
        self.comments = self.comments.reset_index(drop=True)

    def load_metadata(self, filename):
        df = pd.read_csv(filename)
        return df
    
    def parse_urls(self, url):
        videoID = ""
        source = ""
        if "youtube" in url:
            source = "youtube"
            videoID = url.split("watch?v=")[1].split("_channel=")[0].split("&t=")[0].split("&lc=")[0].split("&ab")[0].split("&")[0]
            if len(videoID) != 11:
                print(videoID)
            assert len(videoID) == 11

        elif "bitchute" in url:
            source = "bitchute"
            videoID = url.split("/")[-2]
    
        # return videoID, source
        return  pd.Series({'videoID': videoID, 'source': source})

    def load_comments(self, filename):
        df = pd.read_csv(filename)
        df['label'] = df['label'].apply(lambda x: x == 'yes')
        return df

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = self.comments['comment'][index] if self.args.add_comment else ''
        title = self.comments['title'][index] if self.args.add_title else ''
        desc = self.comments['desc'][index] if self.args.add_description else ''
        transcript = self.comments['transcript'][index] if self.args.add_transcription else ''
        other_comment = self.comments['key_phrases_other_comments'][index] if self.args.add_other_comments else ''

        frame_data = torch.zeros(3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)
        if self.args.add_video:
            frame_data = []
            filename = os.path.join(self.args.video_path, self.comments['source'][index], self.comments['videoID'][index])
            vidcap = cv2.VideoCapture(f"{filename}.mp4")
            total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            frames_step = total_frames // NUM_FRAMES
            for i in range(NUM_FRAMES):
                vidcap.set(1, i * frames_step)
                ret, image = vidcap.read()
                if ret:
                    image = self.transform(image)
                    image = image[None, ...]
                    frame_data.append(image)

            vidcap.release()
            try:
                frame_data = torch.cat(frame_data)
            except:
                print("Error 1: ", comment, frame_data, self.comments['source'][index], self.comments['videoID'][index], filename, total_frames, frames_step)
            
            try:
                frame_data = frame_data.reshape((3, NUM_FRAMES, CROP_SIZE, CROP_SIZE))
            except:
                print("Error 2: ", comment, frame_data.shape)
            # video = EncodedVideo.from_path(f"{filename}.mp4")
            # video_data = video.get_clip(start_sec=0, end_sec=int(video.duration))
            # video_data = self.transform(video_data)
            # frame_data = video_data["video"]
            
        if self.args.multilabel:
            target = np.zeros(5, dtype=float)
            if self.args.remove_none:
               target = np.zeros(4, dtype=float) 
            labels = self.comments['target'][index].split(',')
            for label in labels:
                label = label.strip()
                target[self.mapping[label]] = 1
            target = torch.FloatTensor(target)
        else:
            label = self.comments['label'][index]
            target = torch.FloatTensor([label]) 
        return comment, title, desc, transcript, other_comment, frame_data, target
