import glob
import os
import pathlib
import pickle
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import tensorflow as tf
from nltk.metrics.scores import accuracy, f_measure, precision, recall
from PIL import Image
from skimage import io, transform
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from tqdm import tqdm

BASE_CHECKPOINT = 'ai/models/checkpoints'

def train(model, train_data, val_data=None, epochs=10, callbacks=None):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

def PRFA(predictions, answers):
    pred_indices = [x for x in range(len(predictions)) if predictions[x] == 1]
    actual_indices = [y for y in range(len(answers)) if answers[y] == 1]

    temp_precision = precision(set(pred_indices), set(actual_indices)) # actual labels vs. predicted labels
    temp_recall = recall(set(pred_indices), set(actual_indices))
    temp_f1 = f_measure(set(pred_indices), set(actual_indices))
    temp_accuracy = accuracy(answers, predictions)
    return (temp_precision, temp_recall, temp_f1, temp_accuracy)

def get_preds(dataloader, model):
    preds = []
    labels = []

    for batch in tqdm(dataloader):
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
            if len(batch) > 1:
                labels.extend(batch[1])
        elif isinstance(batch, dict):
            inputs = batch.get('image', batch.get('input'))
            if 'label' in batch:
                labels.extend(batch['label'])
        else:
            inputs = batch

        # Predict on batch
        batch_preds = model.predict_on_batch(inputs)
        preds.extend(batch_preds)

    return np.array(preds), np.array(labels)

def get_clamped_preds(preds, t):
    return (preds > t).astype(int)

def test(preds, labels):
    size = len(preds)
    correct = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            correct += 1

    accuracy = correct / size
    print("Accuracy: ", accuracy)
    return PRFA(preds, labels)

def plot_his(history):
    plt.figure(figsize=(15,12))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = [
        'accuracy',
        'loss',
        'precision',
        'recall',
        'f1'
    ]
    for i, metric in enumerate(metrics):
        plt.subplot(3,2,i+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        if 'val_'+metric in history.history:
            plt.plot(history.epoch, history.history['val_'+metric], color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.show()

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    plt.xticks(np.arange(len(class_names))+.5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names))+.5, class_names, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
