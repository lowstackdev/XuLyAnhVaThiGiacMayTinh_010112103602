# %%
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
PROJECT_ROOT = Path(__file__).parent.resolve()
# PROJECT_ROOT = '/content/drive/MyDrive/Colab Notebooks'
DATASET_DIR = PROJECT_ROOT / "ODIR-5K"
os.chdir(DATASET_DIR)

# %%
FILE_NAME = "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
df = pd.read_excel(FILE_NAME)
print(df.head())

# %%
left_eye_keywords = df["Left-Diagnostic Keywords"].copy()
right_eye_keywords = df["Right-Diagnostic Keywords"].copy()

left_eye_keywords = left_eye_keywords.str.split("，")
right_eye_keywords = right_eye_keywords.str.split("，")

print("Left eye keywords:", len(left_eye_keywords), sum([len(kw) for kw in left_eye_keywords]), [kw for kw in left_eye_keywords])
print("Right eye keywords:", len(right_eye_keywords), sum([len(kw) for kw in left_eye_keywords]), [kw for kw in right_eye_keywords])
