import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import List, Tuple

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_class_distribution_bar(df: pd.DataFrame, label_col: str='label', title: str='Distribution of image count by label'):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=label_col, data=df, hue=label_col, palette='viridis', legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Number of images', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_histogram(df: pd.DataFrame, column: str, title: str = None, bins: int = 30):
    plt.figure(figsize=(12, 6))

    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(data=df, x=column, bins=bins, kde=True, color='teal')
        if title is None:
            title = f'Histogram of {column}'
    else:
        sns.countplot(data=df, x=column, hue=column, palette='viridis', legend=False)
        plt.xticks(rotation=45)
        if title is None:
            title = f'Distribution of {column}'

    plt.title(title, fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_class_distribution_pie(df: pd.DataFrame, label_col: str='label', title: str='Distribution ratio of labels'):
    label_counts=df[label_col].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(label_counts)))
    plt.title(title, fontsize=16)
    plt.axis('equal')
    plt.show()

def _get_image_dimensions(filepaths: List[str]) -> List[Tuple[int, int]]:
    dimensions=[]
    sample_size = min(len(filepaths), 500)
    sample_paths = random.sample(list(filepaths), sample_size)

    for path in sample_paths:
        try:
            with Image.open(path) as img:
                dimensions.append(img.size)
        except Exception as e:
            dimensions.append((None, None))
    return dimensions

def plot_image_dimensions_scatter(df: pd.DataFrame, filepath_col: str='filepath', title: str='Distribution of Image Dimensions (Sampled)'):
    dimensions=_get_image_dimensions(df[filepath_col])
    dims_df=pd.DataFrame(dimensions, columns=['width', 'height']).dropna()

    plt.figure(figsize=(10, 8))
    sns.jointplot(x='width', y='height', data=dims_df, kind='hex', cmap='viridis')
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_random_samples_from_each_class(df: pd.DataFrame, label_col: str='label', filepath_col: str='filepath', num_samples: int=5):
    temp_df = df.copy()
    if temp_df[label_col].dtype == object:
        temp_df['simple_label'] = temp_df[label_col].str.extract(r"'(.*?)'")
    else:
        temp_df['simple_label'] = temp_df[label_col]

    labels=temp_df['simple_label'].unique()
    labels = [l for l in labels if pd.notna(l)]
    num_labels=len(labels)

    fig, axes=plt.subplots(num_labels, num_samples, figsize=(num_samples * 3, num_labels * 3))
    fig.suptitle('Random sample images from each class', fontsize=20)

    for i, label in enumerate(labels):
        sample_df=temp_df[temp_df['simple_label'] == label]

        if len(sample_df) < num_samples:
            image_paths=sample_df[filepath_col].tolist()
        else:
            image_paths=random.sample(sample_df[filepath_col].tolist(), num_samples)

        for j, image_path in enumerate(image_paths):
            ax=axes[i, j] if num_labels > 1 else axes[j]
            try:
                img=Image.open(image_path)
                ax.imshow(img)
                ax.set_title(f'Class: {label}' if j == 0 else "")
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, 'Image load error', ha='center', va='center')
                ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_multi_label_distribution(df: pd.DataFrame, class_cols: List[str], title: str='Distribution of Diagnostic Categories'):
    class_counts = df[class_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='viridis', legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_label_correlation(df: pd.DataFrame, class_cols: List[str], title: str='Correlation Between Diagnostic Categories'):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[class_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_labels_per_sample(df: pd.DataFrame, class_cols: List[str], title: str='Number of Conditions per Patient'):
    num_labels = df[class_cols].sum(axis=1)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=num_labels, hue=num_labels, palette='magma', legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Conditions', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_multilabel_conflict_resolution(df: pd.DataFrame, class_cols: List[str], target_col: str='label', title: str='Target Selection in Multi-label Samples'):
    temp_df = df.copy()
    temp_df['active_count'] = temp_df[class_cols].gt(0).sum(axis=1)

    conflict_df = temp_df[temp_df['active_count'] > 1].copy()

    if conflict_df.empty:
        return

    if conflict_df[target_col].dtype == object:
        conflict_df['simple_label'] = conflict_df[target_col].astype(str).str.extract(r"'(.*?)'")
        conflict_df['simple_label'] = conflict_df['simple_label'].fillna(conflict_df[target_col])
    else:
        conflict_df['simple_label'] = conflict_df[target_col]

    plt.figure(figsize=(12, 6))
    sns.countplot(x='simple_label', data=conflict_df, hue='simple_label', palette='rocket', legend=False, order=conflict_df['simple_label'].value_counts().index)
    plt.title(title, fontsize=16)
    plt.xlabel('Selected Target Label', fontsize=12)
    plt.ylabel('Count of Multi-label Samples', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
