# %%
import os
import re
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("./ODIR-5K/ODIR-5K_Training_Annotations(Updated)_V2.csv")

# %%
print("Total rows:", len(df))
print(df.head())

#  %%
LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
LABEL_COLS = df.columns[7:]

# %%
one_label_rows = df[df[LABEL_COLS].sum(axis=1) == 1].index.to_list()
two_label_rows = df[df[LABEL_COLS].sum(axis=1) == 2].index.to_list()
three_label_rows = df[df[LABEL_COLS].sum(axis=1) == 3].index.to_list()

print("n one label rows:", len(one_label_rows))
print("n two label rows:", len(two_label_rows))
print("n three label rows:", len(three_label_rows))

# %%
def get_single_diagnosis(col_name):
    other_diag_cols = [col for col in LABEL_COLS if col != col_name]

    single_rows = df[(df[col_name] == 1) & (df[other_diag_cols].sum(axis=1) == 0)].index.tolist()

    return single_rows

all_single_diagnosis = [get_single_diagnosis(col) for col in LABELS]
all_single_diagnosis_count = sum(len(diags) for diags in all_single_diagnosis)

for i, col in enumerate(LABELS):
    print(f"{col}:", len(all_single_diagnosis[i]))

print("Total single diagnosis:", all_single_diagnosis_count)

# %%
for i in range(8):
    print(f"Total record with {i + 1} label: {len(df[df[LABEL_COLS].sum(axis=1) == i + 1].index.to_list())}")

# %%
left_eye_diagnosis = df['Left-Diagnostic Keywords'].copy()
right_eye_diagnosis = df['Right-Diagnostic Keywords'].copy()

left_eye_diagnosis = left_eye_diagnosis.str.split(re.compile(r'[,，]'))
right_eye_diagnosis = right_eye_diagnosis.str.split(re.compile(r'[,，]'))

# %%
all_diagnostics = set(
    keyword
    for keywords in pd.concat([left_eye_diagnosis, right_eye_diagnosis]).dropna()
    for keyword in keywords
)

for diags in all_diagnostics:
    print(repr(diags))

print("Total all diagnosis:", len(all_diagnostics))

# %%
from collections import defaultdict, Counter
labels_dict = defaultdict(lambda: defaultdict(int))
final_labels = {}

for _, row in df.iterrows():
    keywords = []
    for col in ["Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]:
        if isinstance(row[col], str):
            kws = re.split(r'[,，]', row[col])
            keywords.extend([kw.strip() for kw in kws if kw.strip()])

    vec = row[LABELS].to_numpy()
    active_idx = np.where(vec == 1)[0]

    for kw in keywords:
        for i in active_idx:
            lab = LABELS[i]
            labels_dict[lab][kw] += 1

        counts = [labels_dict[LABELS[i]][kw] for i in active_idx]
        best_idx = active_idx[np.argmax(counts)]
        best_lab = LABELS[best_idx]

        if kw not in final_labels or labels_dict[best_lab][kw] > final_labels[kw][1]:
            final_labels[kw] = (best_lab, labels_dict[best_lab][kw])

for kw, (lab, count) in final_labels.items():
    print(f"{lab}: {count} {kw}")

total_final_labels = len(final_labels)
print("Total diagnostics:", total_final_labels)
