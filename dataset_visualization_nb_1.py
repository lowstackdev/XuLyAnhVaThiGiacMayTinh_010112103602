# %% [markdown]
# # Dataset Visualization
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from PIL import Image

current_dir = os.getcwd()
project_root = os.path.dirname(os.path.dirname(os.getcwd()))

if current_dir not in sys.path:
    sys.path.append(current_dir)

from data_visualization_1 import (
    plot_class_distribution_bar,
    plot_class_distribution_pie,
    plot_histogram,
    plot_image_dimensions_scatter,
    plot_random_samples_from_each_class,
    plot_multi_label_distribution,
    plot_label_correlation,
    plot_labels_per_sample,
    plot_multilabel_conflict_resolution
)

os.chdir(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)

# %% [markdown]
# ## Load Data

# %%
csv_path = 'ai/input/datasets/kaggle/rohitrawat25/combined-fundus-images/label_images.csv'
df = pd.read_csv(csv_path)

image_dir = 'ai/input/datasets/kaggle/rohitrawat25/combined-fundus-images/images'
df['filepath'] = df['images'].apply(lambda x: os.path.join(image_dir, x))

df.head()

# %% [markdown]
# ## 1. Class Distribution
#
# Visualizing the distribution of the primary labels.

# %%
plot_class_distribution_bar(df, label_col='label')
plot_class_distribution_pie(df, label_col='label')

# %% [markdown]
# ## 2. Demographic Distribution
#
# Visualizing age and sex distributions using the `plot_histogram` function.

# %%
plot_histogram(df, column='Patient Age', title='Distribution of Patient Age', bins=30)
plot_histogram(df, column='Patient Sex', title='Distribution of Patient Sex')

# %% [markdown]
# ## 3. Image Dimensions
#
# Checking the distribution of image sizes (sampled).

# %%
plot_image_dimensions_scatter(df, filepath_col='filepath')

# %% [markdown]
# ## 4. Multi-label Analysis
#
# Analyzing overlapping conditions and correlations between categories.

# %%
class_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
plot_multi_label_distribution(df, class_cols)
plot_label_correlation(df, class_cols)
plot_labels_per_sample(df, class_cols)
plot_multilabel_conflict_resolution(df, class_cols)

# %% [markdown]
# ## 5. Random Samples from Each Class

# %%
plot_random_samples_from_each_class(df, label_col='label', filepath_col='filepath', num_samples=5)
