# %%
import os
import glob
from pathlib import Path
import shutil
import time
from random import sample
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

        CACHE_DIR = PROJECT_ROOT / "cache" / "odir5k_multi_class_classification"

    DATASET_DIR = PROJECT_ROOT / "ODIR-5K"

    # Dataset configuration
    ANNOTATION_FILE_NAME = "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    TRAINING_SOURCE_PATH = "ODIR-5K_Training_Images/"
    TESTING_SOURCE_PATH = "ODIR-5K_Testing_Images/"
    LABEL_STRINGS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
    VALIDATION_FRACTION = 0.1

    # Image processing
    TARGET_SIZE = (230, 230)
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

all_key_single_label = [get_key_diagnosis_single(test_df.columns[7 + i]) for i in range(len(config.LABEL_STRINGS))]
print("All keys:", sum(len(x) for x in all_key_single_label))
for i in range(len(config.LABEL_STRINGS)): print(f"{config.LABEL_STRINGS[i]}: {len(all_key_single_label[i])} | {all_key_single_label[i]}")

# %%
# all_key_sets = [set(keywords) for keywords in all_key_single_label]

# # Remove "normal" keyword from all groups
# normal_keywords = all_key_sets[0]
# all_key_sets[1:-1] = [keywords - normal_keywords for keywords in all_key_sets[1:-1]]

# # Remove duplicate keywords between groups
# for i, current in enumerate(all_key_sets):
#     for next in all_key_sets[i + 1 :]:
#         next -= current & next

# all_key_single_label = [list(keywords) for keywords in all_key_sets]
# print("Total intersected:", sum(len(x) for x in all_key_single_label))
# for i in range(len(config.LABEL_STRINGS)): print(f"{config.LABEL_STRINGS[i]}: {len(all_key_single_label[i])} | {all_key_single_label[i]}")

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
TRAINING_PATH = "training/"
VALIDATION_PATH = "validation/"

for path in [TRAINING_PATH, VALIDATION_PATH]:
    shutil.rmtree(path, ignore_errors=True)
    for label in config.LABEL_STRINGS:
        os.makedirs(os.path.join(path, label), exist_ok=True)

training_source_files = os.listdir(config.TRAINING_SOURCE_PATH)
testing_source_files = os.listdir(config.TESTING_SOURCE_PATH)

print(f"Total training source images: {len(training_source_files)}")
print(f"Total testing source images: {len(testing_source_files)}")

# Group files by patient ID to prevent data leakage (same patient's eyes in different sets)
# File naming convention: [PatientID]_[eye].jpg
from collections import defaultdict

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
    label_mapping = list(zip(all_key_single_label, config.LABEL_STRINGS))

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
                    shutil.copy(source_path + file_name, os.path.join(dest_path, label_dir))
                    break

for files, src, dest, name in [
    (training_files, config.TRAINING_SOURCE_PATH, TRAINING_PATH, "Training"),
    (validation_files, config.TRAINING_SOURCE_PATH, VALIDATION_PATH, "Validation"),
]:
    print(f"\nOrganizing {name} files...")
    organize_eye_images_by_diagnosis(files, src, dest)
    for label in config.LABEL_STRINGS:
        count = len(os.listdir(os.path.join(dest, label)))
        print(f"{name} {label} count: {count}")

# %%
# caching data
def _get_raw_cached_dataset(self: tf.data.Dataset, name) -> tf.data.Dataset:
    cache_dir = config.CACHE_DIR / '_get_raw_cached_dataset'
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = str(cache_dir / f"{name}.cache")
    self = self.cache(cache_path)

    # warmup cache
    if not (cache_dir / f"{name}.cache").exists():
        self.enumerate().reduce(np.int64(0), lambda x, _: x + 1)

    return self

# oversampling data
@tf.function
def _get_balanced_dataset(self: tf.data.Dataset, num_classes=8) -> tf.data.Dataset:
    total_samples = tf.data.experimental.cardinality(self)

    # infinite cardinality case
    total_samples = tf.cond(
        tf.equal(total_samples, tf.data.experimental.INFINITE_CARDINALITY),
        lambda: tf.constant(10000, dtype=tf.int64),
        lambda: total_samples
    )

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

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode=config.COLOR_MODE,
    batch_size=config.BATCH_SIZE,
    image_size=config.TARGET_SIZE,
    shuffle=True,
    seed=42,
    interpolation="lanczos3"
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    labels="inferred",
    label_mode="categorical",
    color_mode=config.COLOR_MODE,
    batch_size=config.BATCH_SIZE,
    image_size=config.TARGET_SIZE,
    shuffle=False,
    interpolation="lanczos3"
)

train_generator = (
    raw_train_ds
    ._get_raw_cached_dataset(name="training")
    .unbatch()
    ._get_balanced_dataset()
    .shuffle(buffer_size=1000)
    .batch(config.BATCH_SIZE, drop_remainder=False)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_generator = (
    raw_val_ds
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
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=0.1, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(factor=0.15, value_range=(0, 1)),
            tf.keras.layers.RandomContrast(factor=0.15),
            tf.keras.layers.GaussianNoise(stddev=0.01),
            tf.keras.layers.RandomZoom(height_factor=(-0.02, 0.02), width_factor=(-0.02, 0.02), fill_mode="nearest"),
        ])

        rescaling_layer = tf.keras.layers.Rescaling(1./255)

        model = tf.keras.models.Sequential([
                tf.keras.Input(shape=config.TARGET_SIZE + config.SHAPE_ADD),
                rescaling_layer,
                augmentation_layers,
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
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(8, activation="softmax")])

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
y_train = np.concatenate([np.argmax(y.numpy(), axis=1) for x, y in raw_train_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)])
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
model.evaluate(validation_generator)

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
        for i, label_col in enumerate(config.LABEL_STRINGS):
            if row_data[LABEL_COLS[i]] == 1:
                true_label = label_col
                break

    test_images.append((os.path.join(config.TESTING_SOURCE_PATH, file_name), true_label))

print(f"\nPredicting {len(test_images)} files from testing set")
print("Class Mapping:", {i: name for i, name in enumerate(raw_train_ds.class_names)})

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
        pred_label = raw_train_ds.class_names[pred_idx]
        x_idx = np.argmax((classes > 0.05).astype("int32"))

        # table row
        filename = os.path.basename(img_path)
        probs_str = ", ".join([f"{prob:.4f}" for prob in classes[0]])
        row = f"| {filename:<30} | {true_label:<15} | {pred_label:<15} | {pred_idx:<8} | {x_idx:<5} | {probs_str:<50} |"
        print(row)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
