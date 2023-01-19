import itertools
import os.path
import random
from http.client import IncompleteRead
from urllib.error import URLError

import numpy as np
import pandas as pd
import torch
import torchvision
from pytube import YouTube
from pytube.exceptions import RegexMatchError, VideoPrivate, VideoUnavailable
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from tqdm import tqdm
from ast import literal_eval


def download_videos(yt_file: str, path_to_download: str, force: bool = False):
    file_table = pd.read_csv(yt_file)
    for i, vid in tqdm(enumerate(file_table['link'].values), total=file_table.shape[0]):
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={vid}")
        except RegexMatchError:
            continue
        try:
            stream = yt.streams.first()
        except VideoPrivate:
            continue
        except VideoUnavailable:
            continue
        except KeyError:
            continue

        dir_to_load = os.path.join(path_to_download, file_table.loc[i, 'category'])
        if not os.path.exists(dir_to_load):
            os.mkdir(dir_to_load)

        e_files = os.listdir(dir_to_load)
        if stream.default_filename in e_files and not force:
            continue

        try:
            stream.download(dir_to_load)
        except IncompleteRead:
            continue
        except URLError:
            continue


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi", ".3gpp")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)


class YouTubeDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(YouTubeDataset).__init__()
        self.samples = get_samples(root)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer

            # Seek and return frames
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output


class EmbeddingsDataset(Dataset):

    def __init__(self, table: pd.DataFrame):
        self.__table = table
        self.indexes = list(self.__table.index)
        print(len(set(self.__table.index)), self.__table.shape)
    #
    # def split_train_test(self, train_part: float = 0.8):
    #     video_ids = list(set(self.__table.loc[:, 'id_video'].values))
    #     train_ids = random.sample(video_ids, k=int(len(video_ids) * train_part))
    #     test_ids = list(set(video_ids) - set(train_ids))
    #     train_indexes = list(self.__table.loc[self.__table.id_video.isin(train_ids)].index)
    #     test_indexes = list(self.__table.loc[self.__table.id_video.isin(test_ids)].index)
    #
    #     train_dataset, test_dataset = EmbeddingsDataset(filepath=self.__filepath), EmbeddingsDataset(filepath=self.__filepath)
    #     train_dataset.indexes = train_indexes
    #     test_dataset.indexes = test_indexes
    #     return train_dataset, test_dataset

    def __getitem__(self, item):
        index = self.indexes[item]
        row = self.__table.loc[index]
        embedding = row.iloc[5:]
        embedding = list(map(float, embedding))
        embedding = torch.tensor(data=embedding)
        prob_vector = row.iloc[1:5]
        return row[0], embedding, torch.tensor(prob_vector.values)

    def __len__(self):
        return len(self.indexes)
