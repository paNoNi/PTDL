import enum
import os
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights
from torchvision import transforms
import torch
from tqdm import tqdm
from enum import Enum

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Status(Enum):
    RandomVideoIsNopOpened = 1
    VideoIsNopOpened = 2


def transformation(frame: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame)


def get_model():
    model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    model.head = torch.nn.Identity()
    return model.to(device)


def get_rand_crop_coords(height, width):
    y_top = random.randint(0, int(height * 0.8))
    y_bot = random.randint(int(height * 0.2), height - y_top) + y_top
    x_top = random.randint(0, int(width * 0.8))
    x_bot = random.randint(int(width * 0.2), width - x_top) + x_top
    return y_top, y_bot, x_top, x_bot


def cut_mix(orig_frame: np.ndarray, adv_frame: np.ndarray):
    while True:
        orig_height, orig_width, _ = orig_frame.shape
        adv_height, adv_width, _ = adv_frame.shape
        orig_y_top, orig_y_bot, orig_x_top, orig_x_bot = get_rand_crop_coords(orig_height, orig_width)
        adv_y_top, adv_y_bot, adv_x_top, adv_x_bot = get_rand_crop_coords(adv_height, adv_width)

        cropped = adv_frame[adv_y_top: adv_y_bot, adv_x_top: adv_x_bot, :]
        try:
            orig_frame[orig_y_top: orig_y_bot, orig_x_top: orig_y_bot, :] = \
                cv2.resize(cropped.astype(np.float32), dsize=(orig_y_bot - orig_x_top, orig_y_bot - orig_y_top))
        except Exception:
            continue

        part = (adv_y_bot - adv_y_top) * (adv_x_bot - adv_x_top) / (orig_height * orig_width)
        break

    return orig_frame, part


def convert_videos_to_embeddings(videos_dir: str, dir_to_load: str, file_name: str, count: int, train_part: float = 0.8):
    labels = [name for name in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, name))]
    print(f'Found directories/labels: {", ".join(labels)}')
    model = get_model()
    full_count = 0
    table = pd.DataFrame(data=[], columns=['category', 'filepath'])
    for label in labels:
        cur_label_videos_dir = os.path.join(videos_dir, label)
        videos = os.listdir(cur_label_videos_dir)
        for video in videos:
            table.loc[full_count, ['category']] = label
            table.loc[full_count, ['filepath']] = os.path.join(cur_label_videos_dir, video)
            full_count += 1

    table = table.sample(frac=1).reset_index(drop=True)
    table = table.head(n=count)
    current_video = 0
    with open(os.path.join(dir_to_load, f'{file_name}_train.csv'), 'a') as train_storage, \
            open(os.path.join(dir_to_load, f'{file_name}_test.csv'), 'a') as test_storage:
        header = 'id_video,prob_1,prob_2,prob_3,prob_4,' + ','.join([str(value) for value in range(1024)]) + '\n'
        train_storage.write(header)
        test_storage.write(header)
        for i in range(table.shape[0]):
            while True:
                video_full_path = table.loc[i, 'filepath']
                label = table.loc[i, 'category']

                try:
                    answer = read_video(model, table, labels, video_full_path, current_video, full_count, label)
                    if answer == Status.RandomVideoIsNopOpened:
                        continue
                    elif answer == Status.VideoIsNopOpened:
                        break
                    elif answer is None:
                        break
                    else:
                        embeddings, labels_prob, is_train = answer
                except IOError:
                    current_video += 1
                    break
                if embeddings.shape[0] == 0:
                    break

                id_emb = np.array([i] * embeddings.shape[0], dtype=np.float)
                id_emb = np.reshape(id_emb, newshape=(-1, 1))
                embeddings = np.concatenate([id_emb, labels_prob, embeddings], axis=1, dtype=np.str)
                if is_train:
                    train_storage.write('\n'.join([','.join(row) for row in embeddings]))
                    train_storage.write('\n')
                else:
                    test_storage.write('\n'.join([','.join(row) for row in embeddings]))
                    test_storage.write('\n')

                current_video += 1
                break


def get_random_frame(table: pd.DataFrame) -> Tuple[np.ndarray, str]:
    while True:
        random_ind = random.randint(0, table.shape[0] - 1)
        rand_label = table.iloc[random_ind].loc['category']
        rand_video_full_path = table.loc[random_ind, 'filepath']
        rand_cap = cv2.VideoCapture(rand_video_full_path)
        if not rand_cap.isOpened():
            continue
        total_frames = rand_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        rand_cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(a=0, b=total_frames - 1))
        rand_ret, rand_image = rand_cap.read()
        if not rand_ret:
            continue
        else:
            return rand_image, rand_label


def read_video(model: torch.nn.Module, table: pd.DataFrame, labels: list, path_to_video: str, cur_video: int, full_count: int, label: str):
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        print(f"Error opening video stream or file: {path_to_video}")
        return Status.VideoIsNopOpened

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(lenght / fps)
    emb_space = list()
    subbar = tqdm(total=duration, position=0)
    labels_space = list()
    is_train = random.random() < 0.8
    current_sec = 0
    if lenght > 18_000:
        return None
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1_000)
        ret, frame = cap.read()
        rand_image, rand_label = get_random_frame(table)

        if ret:
            if not is_train:
                frame, part = frame, .0
            else:
                frame, part = cut_mix(frame, rand_image)

            frame = transformation(frame)
            labels_prob = np.zeros(shape=(1, 4))
            labels_prob[0, labels.index(label)] += 1 - part
            labels_prob[0, labels.index(rand_label)] += part
            labels_space.append(labels_prob)
            current_sec += 1
        else:
            current_sec += 1
            break

        embeddings = model(torch.unsqueeze(frame, dim=0).to(device))[0]
        emb_space.append(embeddings.detach().cpu().numpy())

        subbar.update(1)
        subbar.set_description_str(f'Progress: {cur_video}/{full_count}')
        subbar.set_postfix_str(f'Video category: {label}')
    subbar.clear()

    labels_space = np.concatenate(labels_space, axis=0)
    return np.array(emb_space), labels_space, is_train
