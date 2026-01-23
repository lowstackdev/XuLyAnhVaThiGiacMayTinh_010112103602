import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Set
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

def get_img_paths(path):
    img_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    return img_paths

def get_dataframe(paths, labels):
    p_series = pd.Series(paths, name='filepaths')
    l_series = pd.Series(labels, name='labels')
    return pd.concat([p_series, l_series], axis=1)

def split_data(df: pd.DataFrame):
    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=df['labels'])
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=42, stratify=dummy_df['labels'])

    return train_df, valid_df, test_df

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = dataframe['labels'].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx

def augment_data(train_df, valid_df, test_df, img_size, batch_size=32):
    # Custom testing batch size
    test_length = len(test_df)
    test_batch_size = max(sorted([test_length//n
                                  for n in range(1, test_length + 1)
                                  if test_length%n==0 and test_length/n<=80]))

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=[0.5, 1.5]),
        transforms.ToTensor(),
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = ImageDataset(train_df, transform=train_transform)
    valid_dataset = ImageDataset(valid_df, transform=valid_test_transform)
    test_dataset = ImageDataset(test_df, transform=valid_test_transform)

    # Create data loaders
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_data, valid_data, test_data

def draw_bounding_box(img_path):
    pass
