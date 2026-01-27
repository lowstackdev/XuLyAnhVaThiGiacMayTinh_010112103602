# %%
import os
from pathlib import Path
import shutil
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing import image

print(tf.__version__)

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

# %%
# mlb = MultiLabelBinarizer()

# combined_keywords = pd.concat([left_eye_keywords, right_eye_keywords])
# mlb.fit(combined_keywords)

# all_diagnosis = list(mlb.classes_)
# print("All diagnosis keys:", all_diagnosis)
# print("Total different keys diagnosis:", len(all_diagnosis))

# %%
test_df = df.copy()
LABEL_COLS = test_df.columns[7:]

def get_key_diagnosis_single(col_name):
    # Get other diagnosis columns
    other_diag_cols = [col for col in LABEL_COLS if col != col_name]

    # Find rows where target column == 1 AND all other diagnosis columns == 0
    single_rows = test_df[(test_df[col_name] == 1) & (test_df[other_diag_cols].sum(axis=1) == 0)].index

    unique_keywords = set().union(*[set(left_eye_keywords[row]) | set(right_eye_keywords[row]) for row in single_rows])

    return list(unique_keywords)

LABEL_STRINGS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
all_key_single_label = [get_key_diagnosis_single(test_df.columns[7 + i]) for i in range(len(LABEL_STRINGS))]
print("All keys:", sum(len(x) for x in all_key_single_label))
for i in range(len(LABEL_STRINGS)): print(f"{LABEL_STRINGS[i]}: {len(all_key_single_label[i])} | {all_key_single_label[i]}")

# %%
all_key_sets = [set(keywords) for keywords in all_key_single_label]

# Remove "normal" keyword from all groups
normal_keywords = all_key_sets[0]
all_key_sets[1:-1] = [keywords - normal_keywords for keywords in all_key_sets[1:-1]]

# Remove duplicate keywords between groups
for i, current in enumerate(all_key_sets):
    for next in all_key_sets[i + 1 :]:
        next -= current & next

all_key_single_label = [list(keywords) for keywords in all_key_sets]
print("Total intersected:", sum(len(x) for x in all_key_single_label))
for i in range(len(LABEL_STRINGS)): print(f"{LABEL_STRINGS[i]}: {len(all_key_single_label[i])} | {all_key_single_label[i]}")

# %%
# double_diagnosis_row = test_df[test_df[LABEL_COLS].sum(axis=1) > 1].index.tolist()
# double_diagnosis_row = sorted(set(double_diagnosis_row))
# not_listed = {keyword
#               for row in double_diagnosis_row
#               for keyword in left_eye_keywords[row] + right_eye_keywords[row] if keyword not in all_key_single_label}

# print("Double label row:", len(double_diagnosis_row))
# print("Not listed diagnosis key:", len(not_listed))

# def get_all_recognized_key(all_key):
#     return list(set([keyword for keywords in all_key for keyword in set(keywords)]))

# def intersect_from_multi_label(keyword_groups):
#     known_keywords = set().union(*keyword_groups)
#     unrecognized_keywords = set()

#     for record_idx in double_diagnosis_row:
#         keywords = left_eye_keywords[record_idx] + right_eye_keywords[record_idx]
#         undiscovered = set(kw for kw in keywords if kw not in known_keywords)

#         if undiscovered:
#             related_groups = [
#                 col_idx - 7
#                 for col_idx in range(7, len(test_df.columns))
#                 if test_df.iloc[record_idx, col_idx] == 1
#             ]

#             if len(related_groups) == 1 and len(undiscovered) == 1:
#                 keyword_groups[related_groups[0]].append(undiscovered.pop())
#                 known_keywords.add(keyword_groups[related_groups[0]][-1])
#             else:
#                 unrecognized_keywords.update(undiscovered)

#     return keyword_groups, list(unrecognized_keywords)

# # Process until convergence
# prev_count = 0
# while True:
#     prev_count = len(all_key_diagnosis)
#     all_key_single_label, unrecognized_keywords_list = intersect_from_multi_label(all_key_single_label)
#     all_key_diagnosis = get_all_recognized_key(all_key_single_label)
#     print(unrecognized_keywords_list)
#     if len(all_key_diagnosis) == prev_count:
#         print(True)
#         break

# %%
TRAINING_SOURCE_PATH = "ODIR-5K_Training_Images/"
TESTING_SOURCE_PATH = "ODIR-5K_Testing_Images/"

TRAINING_PATH = "training/"
VALIDATION_PATH = "validation/"

for path in [TRAINING_PATH, VALIDATION_PATH]:
    shutil.rmtree(path, ignore_errors=True)
    for label in LABEL_STRINGS:
        os.makedirs(os.path.join(path, label), exist_ok=True)

# %%
training_source_files = os.listdir(TRAINING_SOURCE_PATH)
testing_source_files = os.listdir(TESTING_SOURCE_PATH)

print(f"Total training source images: {len(training_source_files)}")
print(f"Total testing source images: {len(testing_source_files)}")

VALIDATION_FRACTION = 0.1

# Group files by patient ID to prevent data leakage (same patient's eyes in different sets)
# File naming convention: [PatientID]_[eye].jpg
from collections import defaultdict

patient_to_files = defaultdict(list)
for f in training_source_files:
    patient_id = f.split("_")[0]
    patient_to_files[patient_id].append(f)

unique_patient_ids = list(patient_to_files.keys())
n_val_patients = int(len(unique_patient_ids) * VALIDATION_FRACTION)

# Randomly select patients for each set
all_patient_ids = sample(unique_patient_ids, len(unique_patient_ids))
training_patient_ids = all_patient_ids[n_val_patients:]
validation_patient_ids = all_patient_ids[:n_val_patients]

training_files = [f for pid in training_patient_ids for f in patient_to_files[pid]]
validation_files = [f for pid in validation_patient_ids for f in patient_to_files[pid]]
print(f"Total training files: {len(training_files)}")
print(f"Total validation files: {len(validation_files)}")

# %%
def organize_eye_images_by_diagnosis(file_list, source_path, dest_path):
    "Organize eye images into diagnosis-specific directories based on keywords"
    label_mapping = list(zip(all_key_single_label, LABEL_STRINGS))

    for file_name in file_list:
        # Find matching row in the dataframe
        nrow = None
        keywords_data = None

        # Check if file matches Left-Fundus or Right-Fundus column
        for col, keywords in [("Left-Fundus", left_eye_keywords), ("Right-Fundus", right_eye_keywords)]:
            for i, val in enumerate(df[col]):
                if val == file_name:
                    nrow = i
                    keywords_data = keywords
                    break
            if nrow is not None:
                break

        if nrow is None:
            # If no match found, copy to the first category (Normal) as default
            # shutil.copy(source_path + file_name, os.path.join(dest_path, LABEL_STRINGS[0]))
            continue

        # Find matching diagnosis label
        for key_list, label_dir in label_mapping:
            if any(keyword in key_list for keyword in keywords_data[nrow]):
                shutil.copy(source_path + file_name, os.path.join(dest_path, label_dir))
                break

for files, src, dest, name in [
    (training_files, TRAINING_SOURCE_PATH, TRAINING_PATH, "Training"),
    (validation_files, TRAINING_SOURCE_PATH, VALIDATION_PATH, "Validation"),
]:
    print(f"\nOrganizing {name} files...")
    organize_eye_images_by_diagnosis(files, src, dest)
    for label in LABEL_STRINGS:
        count = len(os.listdir(os.path.join(dest, label)))
        print(f"{name} {label} count: {count}")

# %%
TARGET_SIZE = (200, 300)  # (int(height/16), int(width/16))
COLOR_MODE = "rgb"
COLOR_SHAPE_MAP = {"grayscale": (1,), "rgb": (3,), "rgba": (4,)}
SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

# %%
# 1. Load Raw Datasets
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode=COLOR_MODE,
    batch_size=32,
    image_size=TARGET_SIZE,
    shuffle=True,
    seed=42,
    interpolation="lanczos3")

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode=COLOR_MODE,
    batch_size=32,
    image_size=TARGET_SIZE,
    shuffle=False,
    interpolation="lanczos3")

# 2. Define Preprocessing/Augmentation Pipeline
augmentation_layers = tf.keras.Sequential([
    # 1. GEOMETRIC TRANSFORMATIONS (limited)
    tf.keras.layers.RandomRotation(factor=0.1, fill_mode="nearest"),  # ±36 degrees
    tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode="nearest"),  # zoom
    tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),  # 5%
    # 2. PHOTOMETRIC TRANSFORMATIONS (important)
    tf.keras.layers.RandomBrightness(factor=0.15, value_range=(0, 1)),  # 15%
    tf.keras.layers.RandomContrast(factor=0.15),  # 15% contrast variation
    # 3. NOISE & ARTIFACTS (real-world simulation)
    tf.keras.layers.GaussianNoise(stddev=0.01),  # noise
    # 4. BLUR (simulating focus issues)
    tf.keras.layers.RandomZoom(height_factor=(-0.02, 0.02), width_factor=(-0.02, 0.02), fill_mode="nearest"),  # blur effect
])

rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255)

def prepare_dataset(ds, augment=False):
    # Apply rescaling to all
    ds = ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        # Apply augmentations only to training
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Enable caching and prefetching for high performance
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_generator = prepare_dataset(raw_train_ds, augment=True)
validation_generator = prepare_dataset(raw_val_ds)

# Extract labels from training dataset for class weight calculation
train_labels = np.concatenate([y for x, y in raw_train_ds], axis=0)
train_labels = np.argmax(train_labels, axis=1)  # Convert one-hot to class indices

class_weight_vals = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weight_vals))

# %%
USE_MODEL = "using custom"
USE_PRETRAINED_MODEL = False

INPUT_SHAPE = TARGET_SIZE + SHAPE_ADD
N_EPOCH = 1
LEARNING_RATE = 0.0001
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)  # tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

ACCURACY_SCORE = "accuracy"
PRECISION_SCORE = tf.keras.metrics.Precision(name="precision")
RECALL_SCORE = tf.keras.metrics.Recall(name="recall")
AUC_VALUE = tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", multi_label=True)

MODEL_DIR = PROJECT_ROOT / "Trained_Models" / "ODIR5K-Multi-Class"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_WEIGHTS = str(MODEL_DIR / "ODIR5K_weights.weights.h5")
MODEL_SAVE_FINAL = str(MODEL_DIR / "ODIR5K_final.keras")
CHECKPOINT_PATH = str(MODEL_DIR / "ODIR5K.keras")

callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_auc", save_best_only=True, mode="max", verbose=1),
             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)]

# %%
if os.path.isfile(str(MODEL_SAVE_FINAL)) and USE_PRETRAINED_MODEL:
    print("Using saved model")
    model = tf.keras.models.load_model(str(MODEL_SAVE_FINAL))
else:
    print("No using saved model")
    if USE_MODEL == "using custom":
        def create_conv_block(filters, kernel_size=(3, 3), activation="relu"):
            return [
                tf.keras.layers.Conv2D(filters, kernel_size, activation=activation),
                tf.keras.layers.Conv2D(filters, kernel_size, activation=activation),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.BatchNormalization()]

        model = tf.keras.models.Sequential([
                tf.keras.Input(shape=INPUT_SHAPE),
                *create_conv_block(32),
                *create_conv_block(64),
                *create_conv_block(128),
                *create_conv_block(256),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(8, activation="softmax")])

model.summary(line_length=100)
model.compile(loss="categorical_crossentropy",
              optimizer=OPTIMIZER,
              metrics=[ACCURACY_SCORE, PRECISION_SCORE, RECALL_SCORE, AUC_VALUE])

# %%
history = model.fit(train_generator=train_generator,
                    validation_data=validation_generator,
                    epochs=N_EPOCH,
                    verbose=1,
                    class_weight=class_weights,
                    callbacks=callbacks)

# %%
model.save_weights(MODEL_SAVE_WEIGHTS)
model.save(MODEL_SAVE_FINAL)

# %%
metrics = [
    ("accuracy", "accuracy", 0),
    ("loss", "loss", 1),
    ("precision", "Precision", 2),
    ("auc", "AUC value", 3),
    ("recall", "Recall", 4)
]
epochs = range(1, len(history.history["accuracy"]) + 1)
for key, label, loc in metrics:
    plt.plot(epochs, history.history[key], "r", label=f"Training {label}")
    plt.plot(epochs, history.history[f"val_{key}"], "y", label=f"Validation {label}")
    plt.title(f"Training and validation {label}")
    plt.legend(loc=loc)
    plt.figure()
plt.show()

# %%
model.evaluate(validation_generator)

# %%
test_images = []
for label in LABEL_STRINGS:
    label_dir = os.path.join(TESTING_SOURCE_PATH, label)
    if os.path.exists(label_dir):
        test_images.extend((os.path.join(label_dir, f), label) for f in os.listdir(label_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))

print(f"\nPredicting {len(test_images)} files")

# table header
header = f"| {'File':<30} | {'True Label':<15} | {'Predicted':<15} | {'Pred ID':<8} | {'X ID':<5} | {'Probabilities':<50} |"
separator = f"| {'-'*30} | {'-'*15} | {'-'*15} | {'-'*8} | {'-'*5} | {'-'*50} |"

print(header)
print(separator)

for img_path, true_label in test_images:
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    classes = model.predict(img_array, batch_size=8, verbose=0)
    pred_idx = np.argmax(classes)
    pred_label = raw_train_ds.class_names[pred_idx]
    x_idx = np.argmax((classes > 0.05).astype("int32"))

    # table row
    filename = os.path.basename(img_path)
    probs_str = ", ".join([f"{prob:.4f}" for prob in classes[0]])
    row = f"| {filename:<30} | {true_label:<15} | {pred_label:<15} | {pred_idx:<8} | {x_idx:<5} | {probs_str:<50} |"
    print(row)
