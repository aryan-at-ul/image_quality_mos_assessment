# load_data.py

import torch
import numpy as np
import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import os
from PIL import Image
import json
import random
import re

class TIDLoader(Dataset):
    def __init__(self, path_to_imgs, path_to_json, transform):
        self.path_to_json = path_to_json
        self.path_to_imgs = path_to_imgs
        self.transform = transform
        f = json.load(open(path_to_json))
        self.image_ids = [i['image_id'] for i in f]
        self.labels = [i['label'] for i in f]
        self.distortion_labels = []

        # Extract distortion types from image filenames
        for img_id in self.image_ids:
            # Parse distortion type from the filename
            # Filename format: IXX_YY_Z.bmp
            # Example: I06_03_1.bmp
            match = re.match(r'[iI]\d{2}_(\d{2})_\d', img_id)
            if match:
                distortion_code = match.group(1)
                # Convert distortion_code to integer label (0-based indexing)
                distortion_label = int(distortion_code) - 1  # Distortion codes are from '01' to '24'
                self.distortion_labels.append(distortion_label)
            else:
                raise ValueError(f"Filename {img_id} does not match the expected format.")

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.path_to_imgs, img_id + '.bmp')
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        distortion_label = self.distortion_labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        distortion_label = torch.tensor(distortion_label, dtype=torch.long)
        img, label = self.transform(img, label)
        data = {
            'img': img,
            'label': label,
            'distortion_label': distortion_label
        }
        return data

    def __len__(self):
        return len(self.image_ids)



def build_dataset(path_to_imgs, path_to_json_train, path_to_json_test):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CustomCrop(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    TID_train = TIDLoader(path_to_imgs, path_to_json_train, data_transforms['train'])
    TID_test = TIDLoader(path_to_imgs, path_to_json_test, data_transforms['val'])

    # Split the training data into train and validation sets
    train_len = len(TID_train)
    idx = list(range(train_len))
    random.shuffle(idx)
    split_idx = idx[:int(0.76 * train_len)]
    train_split = Subset(TID_train, split_idx)

    split_idx = idx[int(0.76 * train_len):]
    train_val_split = Subset(TID_train, split_idx)
    val_split = ConcatDataset([train_val_split, TID_test])

    val_len = len(val_split)
    val_idx = list(range(val_len))
    random.shuffle(val_idx)
    val_split_idx = val_idx[:int(0.75 * val_len)]
    final_val_split = Subset(val_split, val_split_idx)

    test_split_idx = val_idx[int(0.75 * val_len):]
    test_split = Subset(val_split, test_split_idx)
    return train_split, final_val_split, test_split


if __name__ == '__main__':
    path_to_json_train = 'tid_labels_train.json'
    path_to_json_test = 'tid_labels_test.json'
    path_to_imgs = '/Users/aryansingh/Downloads/tid2013/distorted_images/'
    TID_train, TID_val, TID_test = build_dataset(path_to_imgs, path_to_json_train, path_to_json_test)
    print(f'Training samples: {len(TID_train)}')
    print(f'Validation samples: {len(TID_val)}')
    print(f'Test samples: {len(TID_test)}')
