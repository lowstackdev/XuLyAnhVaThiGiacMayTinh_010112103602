# %%
import os
import glob
from pathlib import Path
from collections import Counter, defaultdict
import shutil
import time
from random import sample
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
import tensorflow as tf

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

print(tf.__version__)

# %%
class Config:
    # Project paths
    try:
        import google.colab
        PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks')
        CACHE_DIR = Path("/content/cache")
        TRAINING_PATH = "/content/training"
        VALIDATION_PATH = "/content/validation"
    except ImportError:
        # Fallback for non-Colab environments where __file__ might be defined
        # or for local development. We need a way to get the current script's path.
        # If running in a local script, __file__ would be defined.
        # If running in a local interactive environment where __file__ isn't defined,
        # we might need to assume the current working directory.
        if '__file__' in locals() or '__file__' in globals():
            PROJECT_ROOT = Path(__file__).parent.resolve()
        else:
            PROJECT_ROOT = Path(os.getcwd())

        CACHE_DIR = Path(f"{PROJECT_ROOT}/cache/odir5k_multi_class_classification")
        TRAINING_PATH = "training"
        VALIDATION_PATH = "validation"

    DATASET_DIR = PROJECT_ROOT / "ODIR-5K"

    # Dataset configuration
    ANNOTATION_FILE_NAME = "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    TRAINING_SOURCE_PATH = "ODIR-5K_Training_Images"
    TESTING_SOURCE_PATH = "ODIR-5K_Testing_Images"
    LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    VALIDATION_FRACTION = 0.1

    # Image processing
    TARGET_SIZE = (512, 512)
    COLOR_MODE = "rgb"
    COLOR_SHAPE_MAP = {"grayscale": (1,), "rgb": (3,), "rgba": (4,)}
    SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

    # Model configuration
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Metrics
    ACCURACY = "accuracy"
    PRECISION = tf.keras.metrics.Precision(name="precision")
    RECALL = tf.keras.metrics.Recall(name="recall")
    AUC_VALUE = tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", multi_label=True)
    LOSS = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25, label_smoothing=0.1)
    F1_SCORE = tf.keras.metrics.F1Score(average='macro', name='f1_score')

    # Model paths
    MODEL_DIR = PROJECT_ROOT / "Trained_Models" / "ODIR5K-Multi-Class"
    MODEL_SAVE_WEIGHTS = str(MODEL_DIR / "ODIR5K_weights.weights.h5")
    MODEL_SAVE_FINAL = str(MODEL_DIR / "ODIR5K_final.keras")
    CHECKPOINT_PATH = str(MODEL_DIR / "ODIR5K.keras")

    # Model selection
    USE_MODEL = "using custom"
    USE_PRETRAINED_MODEL = False

config = Config()

# %%
os.chdir(config.DATASET_DIR)

# %%
df = pd.read_excel(config.ANNOTATION_FILE_NAME)
print(df.head())

# %%
left_eye_keywords = df["Left-Diagnostic Keywords"].copy()
right_eye_keywords = df["Right-Diagnostic Keywords"].copy()

left_eye_keywords = left_eye_keywords.str.split(re.compile(r'[,，]'))
right_eye_keywords = right_eye_keywords.str.split(re.compile(r'[,，]'))

# %%
labels_dict = defaultdict(Counter)
all_diagostic_keywords = [[] for _ in range(len(config.LABELS))]
keyword_label_map = {}

for _, row in df.iterrows():
    keywords = []
    for col in ["Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]:
        if isinstance(row[col], str):
            kws = re.split(r'[,，]', row[col])
            keywords.extend([kw.strip() for kw in kws if kw.strip()])

    vec = row[config.LABELS].to_numpy()
    active_idx = np.where(vec == 1)[0]

    for kw in keywords:
        for i in active_idx:
            lab = config.LABELS[i]
            labels_dict[lab][kw] += 1

        counts = [labels_dict[config.LABELS[i]][kw] for i in active_idx]
        best_idx = active_idx[np.argmax(counts)]
        best_lab = config.LABELS[best_idx]

        if kw not in keyword_label_map or labels_dict[best_lab][kw] > labels_dict[config.LABELS[keyword_label_map[kw]]][kw]:
            if kw in keyword_label_map:
                old_label_index = keyword_label_map[kw]
                if kw in all_diagostic_keywords[old_label_index]:
                    all_diagostic_keywords[old_label_index].remove(kw)

            keyword_label_map[kw] = best_idx
            all_diagostic_keywords[best_idx].append(kw)

all_diagostic_keywords = [list(set(keywords)) for keywords in all_diagostic_keywords]

for kws in all_diagostic_keywords:
    print(kws)
print(sum(len(kws) for kws in all_diagostic_keywords))

# %%
for path in [config.TRAINING_PATH, config.VALIDATION_PATH]:
    shutil.rmtree(path, ignore_errors=True)
    for label in config.LABELS:
        os.makedirs(os.path.join(path, label), exist_ok=True)

training_source_files = os.listdir(config.TRAINING_SOURCE_PATH)
testing_source_files = os.listdir(config.TESTING_SOURCE_PATH)

print(f"Total training source images: {len(training_source_files)}")
print(f"Total testing source images: {len(testing_source_files)}")

# Group files by patient ID to prevent data leakage (same patient's eyes in different sets)
# File naming convention: [PatientID]_[eye].jpg
patient_to_files = defaultdict(list)
for f in training_source_files:
    patient_id = f.split("_")[0]
    patient_to_files[patient_id].append(f)

unique_patient_ids = list(patient_to_files.keys())
n_val_patients = int(len(unique_patient_ids) * config.VALIDATION_FRACTION)

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
    """Organize eye images into diagnosis-specific directories based on keywords"""
    label_mapping = list(zip(all_diagostic_keywords, config.LABELS))

    for file_name in file_list:
        # matching row in the dataframe
        nrow = None
        keywords_data = None

        # if file matches Left-Fundus or Right-Fundus column
        for col, keywords in [("Left-Fundus", left_eye_keywords), ("Right-Fundus", right_eye_keywords)]:
            for i, val in enumerate(df[col]):
                if val == file_name:
                    nrow = i
                    keywords_data = keywords
                    break
            if nrow is not None:
                break

        if nrow is None:
            continue

        # matching diagnosis label
        for keyword in keywords_data[nrow]:
            for key_list, label_dir in label_mapping:
                if keyword in key_list:
                    src_path = os.path.join(source_path, file_name)
                    dest_full_path = os.path.join(dest_path, label_dir, file_name)
                    shutil.copy(src_path, dest_full_path)
                    break

for files, src, dest, name in [
    (training_files, config.TRAINING_SOURCE_PATH, config.TRAINING_PATH, "Training"),
    (validation_files, config.TRAINING_SOURCE_PATH, config.VALIDATION_PATH, "Validation"),
]:
    print(f"\nOrganizing {name} files...")
    organize_eye_images_by_diagnosis(files, src, dest)
    for label in config.LABELS:
        count = len(os.listdir(os.path.join(dest, label)))
        print(f"{name} {label} count: {count}")

# %%
# caching data
def _get_raw_cached_dataset(self: tf.data.Dataset, name) -> tf.data.Dataset:
    cache_dir = config.CACHE_DIR / '_get_raw_cached_dataset'
    cache_dir.mkdir(parents=True, exist_ok=True)

    for lockfile in cache_dir.glob("*.lockfile"):
        try: lockfile.unlink(missing_ok=True)
        except Exception: pass

    cache_path = str(cache_dir / f"{name}.cache")
    self = self.cache(cache_path)

    # warmup cache
    if not (cache_dir / f"{name}.cache").exists():
        self.enumerate().reduce(np.int64(0), lambda x, _: x + 1)

    return self

# oversampling data
def _get_balanced_dataset(self: tf.data.Dataset, num_classes=8) -> tf.data.Dataset:
    total_samples = self.reduce(np.int64(0), lambda x, _: x + 1)

    class_datasets = []
    for i in range(num_classes):
        class_ds = self.filter(lambda x, y: tf.argmax(y) == i).repeat()
        class_datasets.append(class_ds)

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0/num_classes] * num_classes,
        stop_on_empty_dataset=False
    )

    balanced_ds = balanced_ds.take(total_samples)
    return balanced_ds

# extension methods
tf.data.Dataset._get_raw_cached_dataset = _get_raw_cached_dataset
tf.data.Dataset._get_balanced_dataset = _get_balanced_dataset

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def resize_image(image):
    image = tf.image.resize_with_pad(
        image, config.TARGET_SIZE[0],
        config.TARGET_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR
    )
    image.set_shape([config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3])
    return image

def crop_image(image):
    mask = tf.reduce_sum(image, axis=-1) > 10
    non_zero_coords = tf.where(mask)

    if tf.shape(non_zero_coords)[0] == 0:
        return image

    y_min = tf.cast(tf.reduce_min(non_zero_coords[:, 0]), tf.int32)
    y_max = tf.cast(tf.reduce_max(non_zero_coords[:, 0]), tf.int32)
    x_min = tf.cast(tf.reduce_min(non_zero_coords[:, 1]), tf.int32)
    x_max = tf.cast(tf.reduce_max(non_zero_coords[:, 1]), tf.int32)

    image = tf.image.crop_to_bounding_box(image, y_min, x_min, y_max - y_min + 1, x_max - x_min + 1)
    return image

def CLAHE(image):
    # uint8 format (0-255)
    image = tf.cast(image, tf.uint8)
    image_shape = image.shape

    # input numpy array
    image = tf.numpy_function(func=clahe_cv2, inp=[image], Tout=tf.uint8)

    # Reset shape
    image.set_shape(image_shape)
    return image

def clahe_cv2(image):
    # input numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels + convert back to RGB
    lab = cv2.merge((l, a, b))
    image_res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image_res

def get_label_from_dir(file_path):
    normalized_path = tf.strings.regex_replace(file_path, "\\\\", "/")
    parts = tf.strings.split(normalized_path, "/")
    label_name = parts[-2]
    labels_tensor = tf.constant(config.LABELS)
    one_hot = tf.equal(label_name, labels_tensor)
    return tf.cast(one_hot, tf.float32)

raw_train_ds = tf.data.Dataset.list_files(f"{config.TRAINING_PATH}/*/*.jpg")
raw_val_ds = tf.data.Dataset.list_files(f"{config.VALIDATION_PATH}/*/*.jpg")

train_generator = (
    raw_train_ds
    .map(lambda path: (load_image(path), get_label_from_dir(path)), num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda img, lbl: (crop_image(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda img, lbl: (resize_image(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda img, lbl: (CLAHE(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    ._get_raw_cached_dataset(name="training")
    ._get_balanced_dataset()
    .shuffle(buffer_size=1000)
    .batch(config.BATCH_SIZE, drop_remainder=False)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_generator = (
    raw_val_ds
    .map(lambda path: (load_image(path), get_label_from_dir(path)), num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda img, lbl: (crop_image(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda img, lbl: (resize_image(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda img, lbl: (CLAHE(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    ._get_raw_cached_dataset(name="validation")
    .batch(config.BATCH_SIZE, drop_remainder=False)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# %%
if os.path.isfile(str(config.MODEL_SAVE_FINAL)) and config.USE_PRETRAINED_MODEL:
    print("Using saved model")
    model = tf.keras.models.load_model(str(config.MODEL_SAVE_FINAL))
else:
    print("No using saved model")
    if config.USE_MODEL == "using custom":
        model = tf.keras.models.Sequential([
                tf.keras.Input(shape=config.TARGET_SIZE + config.SHAPE_ADD),
                tf.keras.layers.Rescaling(1./255),

                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(factor=0.05, fill_mode="constant", fill_value=0.0),
                tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="constant"),
                tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1)),
                tf.keras.layers.RandomContrast(factor=0.2),

                # Block 1
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.BatchNormalization(),
                # Block 2
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.BatchNormalization(),
                # Block 3
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.BatchNormalization(),
                # Block 4
                tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
                tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.BatchNormalization(),
                # Classifier
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(8, activation="softmax")
        ])

model.summary(line_length=100)
model.compile(
    loss=config.LOSS,
    optimizer=config.OPTIMIZER,
    metrics=[config.ACCURACY, config.PRECISION, config.RECALL, config.AUC_VALUE, config.F1_SCORE]
)

# %%
import gc
gc.collect()

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Calculate class weights from raw training data to handle class imbalance
y_train = np.array([np.argmax(get_label_from_dir(path).numpy()) for path in raw_train_ds])
class_weights = compute_class_weight('balanced', classes=np.arange(8), y=y_train)
class_weight = dict(enumerate(class_weights))

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(config.CHECKPOINT_PATH, monitor="val_auc", save_best_only=True, mode="max", verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.EPOCHS,
    verbose=1,
    class_weight=class_weight,
    callbacks=callbacks,
)

# %%
model.save_weights(config.MODEL_SAVE_WEIGHTS)
model.save(config.MODEL_SAVE_FINAL)

# %%
model.evaluate(validation_generator)

# %%
metrics = [
    ("accuracy", "accuracy"),
    ("loss", "loss"),
    ("precision", "Precision"),
    ("auc", "AUC value"),
    ("recall", "Recall"),
]
epochs = range(1, len(history.history["accuracy"]) + 1)
for key, label in metrics:
    plt.plot(epochs, history.history[key], "r", label=f"Training {label}")
    plt.plot(epochs, history.history[f"val_{key}"], "y", label=f"Validation {label}")
    plt.title(f"Training and validation {label}")
    plt.legend()
    plt.figure()
plt.show()

# %%
test_images = []

filename_to_row_idx = {
    fname: idx
    for idx, row in df.iterrows()
    for fname in [row['Left-Fundus'], row['Right-Fundus']]
}

for file_name in testing_source_files:
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    idx = filename_to_row_idx.get(file_name)
    true_label = "Unknown"

    if idx is not None:
        row_data = df.iloc[idx]
        for i, label_col in enumerate(config.LABELS):
            if row_data[df.columns[7:][i]] == 1:
                true_label = label_col
                break

    test_images.append((os.path.join(config.TESTING_SOURCE_PATH, file_name), true_label))

print(f"\nPredicting {len(test_images)} files from testing set")
print("Class Mapping:", {i: name for i, name in enumerate(config.LABELS)})

# table header
header = f"| {'File':<30} | {'True Label':<15} | {'Predicted':<15} | {'Pred ID':<8} | {'X ID':<5} | {'Probabilities':<50} |"
separator = f"| {'-'*30} | {'-'*15} | {'-'*15} | {'-'*8} | {'-'*5} | {'-'*50} |"

print(header)
print(separator)

for img_path, true_label in test_images:
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=config.TARGET_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        classes = model.predict(img_array, batch_size=1, verbose=0)
        pred_idx = np.argmax(classes)
        pred_label = config.LABELS[pred_idx]
        x_idx = np.argmax((classes > 0.05).astype("int32"))

        # table row
        filename = os.path.basename(img_path)
        probs_str = ", ".join([f"{prob:.4f}" for prob in classes[0]])
        row = f"| {filename:<30} | {true_label:<15} | {pred_label:<15} | {pred_idx:<8} | {x_idx:<5} | {probs_str:<50} |"
        print(row)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
