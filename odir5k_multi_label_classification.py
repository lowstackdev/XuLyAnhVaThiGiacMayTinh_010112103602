# %%
import os
import glob
from pathlib import Path
import shutil
import time
from random import sample
import uuid
import concurrent.futures
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import cv2
import tensorflow as tf

print(tf.__version__)

# %%
class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.resolve()
    try:
        import google.colab
        PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks')
        CACHE_DIR = Path("/content/cache")
    except ImportError:
        CACHE_DIR = PROJECT_ROOT / "cache" / "odir5k_multi_label_classification"

    DATASET_DIR = PROJECT_ROOT / "ODIR-5K"

    # Dataset configuration
    ANNOTATION_FILE_NAME = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
    TRAINING_SOURCE_PATH = 'ODIR-5K_Training_Images/'
    TESTING_SOURCE_PATH = 'ODIR-5K_Testing_Images/'
    LABEL_STRINGS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
    VALIDATION_FRACTION = 0.1

    # Image processing
    TARGET_SIZE = (230, 230)
    COLOR_MODE = 'rgb'
    COLOR_SHAPE_MAP = {'grayscale': (1,), 'rgb': (3,), 'rgba': (4,)}
    SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

    # Model configuration
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    LOSS = "binary_crossentropy"
    OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Metrics
    ACCURACY = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
    AUC_VALUE = tf.keras.metrics.AUC(name='auc_value', curve='ROC', summation_method='interpolation', multi_label=True)
    PRECISION = tf.keras.metrics.Precision(thresholds=0.5, name='precision')
    RECALL = tf.keras.metrics.Recall(thresholds=0.5, name='recall')

    # Model paths
    MODEL_DIR = PROJECT_ROOT / "Trained_Models" / "ODIR-5K-Multi-Label"
    MODEL_SAVE_WEIGHTS = str(MODEL_DIR / 'ODIR5K_weights.weights.h5')
    MODEL_SAVE_FINAL = str(MODEL_DIR / 'ODIR5K_final.keras')
    CHECKPOINT_PATH = str(MODEL_DIR / 'ODIR5K.keras')

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
left_eye_keywords = df['Left-Diagnostic Keywords'].copy()
right_eye_keywords = df['Right-Diagnostic Keywords'].copy()

left_eye_keywords = left_eye_keywords.str.split("，")
right_eye_keywords = right_eye_keywords.str.split("，")

# %%
# mlb = MultiLabelBinarizer()

# combined_keywords = pd.concat([left_eye_keywords, right_eye_keywords])
# mlb.fit(combined_keywords)

# all_diagnosis = list(mlb.classes_)
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
# all_key_sets[1:] = [keywords - normal_keywords for keywords in all_key_sets[1:]]

# # Remove duplicate keywords between groups
# for i, current in enumerate(all_key_sets):
#     for next in all_key_sets[i + 1 :]:
#         next -= current & next

# all_key_single_label = [list(keywords) for keywords in all_key_sets]
# print("Intersected:", sum(len(x) for x in all_key_single_label))
# for i in range(len(config.LABEL_STRINGS)): print(f"{config.LABEL_STRINGS[i]}: {len(all_key_single_label[i])} | {all_key_single_label[i]}")

# %%
# %%
# double_diagnosis_row = test_df[test_df[LABEL_COLS].sum(axis=1) > 1].index.tolist()
# double_diagnosis_row = sorted(set(double_diagnosis_row))

# not_listed = {
#     keyword
#     for row in double_diagnosis_row
#     for keyword in left_eye_keywords[row] + right_eye_keywords[row]
#     if keyword not in all_key_single_label
# }

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
#             related_groups = [col_idx - 7 for col_idx in range(7, len(test_df.columns)) if test_df.iloc[record_idx, col_idx] == 1]

#         if len(related_groups) == 1 and len(undiscovered) == 1:
#             keyword_groups[related_groups[0]].append(undiscovered.pop())
#             known_keywords.add(keyword_groups[related_groups[0]][-1])
#         else:
#             unrecognized_keywords.update(undiscovered)

#     return keyword_groups, list(unrecognized_keywords)

# # Process until convergence
# prev_count = 0
# while True:
#   prev_count = len(all_key_diagnosis)
#   all_key_single_label, unrecognized_keywords_list = intersect_from_multi_label(all_key_single_label)
#   all_key_diagnosis = get_all_recognized_key(all_key_single_label)
#   print(unrecognized_keywords_list)
#   if len(all_key_diagnosis) == prev_count:
#     print(True)
#     break

# %%

# %%
def resize_image(image_path, dim):
    img = cv2.imread(image_path)
    if img.shape[1] != img.shape[0]:
        x = img.shape[1] // 2
        y = img.shape[0] // 2
        x -= y
        img = img[0:0 + img.shape[0], x:x + img.shape[0]]
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def CLAHE(image_path, dim, clipLimit, tileGridSize):
    img = resize_image(image_path, dim)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# %%
# Function for generate label to single image

# Return index in key of all diagnosis list
def get_index_label(key, all_key):
    return next((i for i, keywords in enumerate(all_key) if key in keywords), -1)

# Return multilabel by index
def get_multi_label_from_keys(idx_label):
    return [1 if i in idx_label else 0 for i in range(len(config.LABEL_STRINGS))]

def process_fundus_image_with_clahe(img_path, keywords, all_key, target_size):
    """Process a single fundus image with CLAHE enhancement and generate diagnostic labels"""
    # check imgage valid
    if (fundus_img := cv2.imread(img_path)) is None:
        return None, None, None

    # Process keywords to generate multi-label diagnosis
    indices = [get_index_label(key, all_key) for key in keywords]
    indices = list(set(indices))
    label = get_multi_label_from_keys(indices)

    # CLAHE enhancement
    clahe_img = CLAHE(img_path, target_size, 20, (10,10))

    return label, os.path.basename(img_path), clahe_img

def process_patient_record(row_idx, df, left_eye_keywords, right_eye_keywords, all_key, target_size):
    """Process a single patient record (both eyes) in parallel and generate diagnostic data"""
    results = []
    for fundus_col, keywords_col in [('Left-Fundus', left_eye_keywords), ('Right-Fundus', right_eye_keywords)]:
        img_path = os.path.join(config.TRAINING_SOURCE_PATH, df[fundus_col][row_idx])
        label, feature, clahe = process_fundus_image_with_clahe(img_path, keywords_col[row_idx], all_key, target_size)
        if label is not None:
            results.append((label, feature, clahe))

    return results

# Optimized parallel processing
synthetic_labels = []
synthetic_features = []
clahe_images = []

# Parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    process_func = partial(
        process_patient_record,
        df=df,
        left_eye_keywords=left_eye_keywords,
        right_eye_keywords=right_eye_keywords,
        all_key=all_key_single_label,
        target_size=config.TARGET_SIZE
    )

    futures = [executor.submit(process_func, i) for i in range(len(df))]

    for future in concurrent.futures.as_completed(futures):
        for label, feature, clahe_img in future.result():
            if label is not None:
                synthetic_labels.append(label)
                synthetic_features.append(feature)
                clahe_images.append(clahe_img)

# %%
clahe_images = np.stack(clahe_images, axis=0)
synthetic_labels = np.asarray(synthetic_labels)

# Grouping by patient ID to prevent data leakage (same patient's eyes in different sets)
groups = [f.split('_')[0] for f in synthetic_features]

gss = GroupShuffleSplit(n_splits=1, test_size=config.VALIDATION_FRACTION, random_state=1)
train_idx, val_idx = next(gss.split(clahe_images, synthetic_labels, groups=groups))

training_features = clahe_images[train_idx]
training_labels = synthetic_labels[train_idx]
training_filenames = [synthetic_features[i] for i in train_idx]

validation_features = clahe_images[val_idx]
validation_labels = synthetic_labels[val_idx]
validation_filenames = [synthetic_features[i] for i in val_idx]

print("n training:", len(training_filenames))
print("n validation:", len(validation_filenames))

del clahe_images
del synthetic_labels

# %%
# def display_image_samples(features, title, color_mode, target_size):
#     """Display a 2x5 grid of image samples with proper coloring based on color mode"""
#     f, ax = plt.subplots(2, 5)
#     f.set_size_inches(10, 10)
#     f.suptitle(title, fontsize=16)

#     for idx in range(10):
#         i, j = divmod(idx, 5)
#         if color_mode == 'rgb':
#             ax[i,j].imshow(features[idx].reshape(target_size[0], target_size[1], 3), cmap="hsv")
#         else:
#             ax[i,j].imshow(features[idx].reshape(target_size[0], target_size[1]), cmap="gray")

#     plt.tight_layout()
#     plt.show()

# display_image_samples(training_features, "Training Image Samples", COLOR_MODE, TARGET_SIZE)
# display_image_samples(validation_features, "Validation Image Samples", COLOR_MODE, TARGET_SIZE)

# %%
BATCH_SIZE = 32
raw_train_ds = tf.data.Dataset.from_tensor_slices((training_features, training_labels))
raw_val_ds = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))

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

rescaling_layer = tf.keras.layers.Rescaling(1./255)

def preprocess_raw_dataset(ds, name):
    ds = ds.map(lambda x, y: (rescaling_layer(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)

    cache_dir = config.CACHE_DIR / 'preprocess_raw_dataset'
    cache_dir.mkdir(parents=True, exist_ok=True)

    for lockfile in cache_dir.glob(f"{name}*.lockfile"):
        try: os.remove(lockfile)
        except: pass

    cache_path = str(cache_dir / name)
    ds = ds.cache(cache_path)

    ds.ignore_errors().prefetch(tf.data.AUTOTUNE).enumerate().reduce(np.int64(0), lambda x, _: x + 1)

    return ds

def get_balanced_train_dataset(cached_ds, num_classes=8):
    class_datasets = []

    for i in range(num_classes):
        class_ds = cached_ds.filter(lambda x, y: y[i] == 1).repeat()
        class_datasets.append(class_ds)

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0/num_classes] * num_classes,
        stop_on_empty_dataset=False
    )

    total_samples = len(training_features)
    balanced_ds = balanced_ds.take(total_samples)
    return balanced_ds

# cached datasets
train_ds_cached = preprocess_raw_dataset(raw_train_ds, name="training")
val_ds_cached = preprocess_raw_dataset(raw_val_ds, name="validation")

# oversampling training data
balanced_train_ds = get_balanced_train_dataset(train_ds_cached)
train_generator = (
    balanced_train_ds
    .shuffle(buffer_size=500)
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(lambda x, y: (augmentation_layers(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_generator = (
    val_ds_cached
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# %%
if os.path.isfile(str(config.MODEL_SAVE_FINAL)) and config.USE_PRETRAINED_MODEL:
    print("Using saved model")
    model = tf.keras.models.load_model(str(config.MODEL_SAVE_FINAL))
else:
    print("No using saved model")
    if config.USE_MODEL == "using custom":
        inputs = tf.keras.Input(shape=config.TARGET_SIZE + config.SHAPE_ADD)

        # Conv larger kernel
        x = tf.keras.layers.Conv2D(32, (7,7), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)

        # Block 1
        shortcut_64 = tf.keras.layers.Conv2D(64, (1,1), padding='same')(x) if x.shape[-1] != 64 else x
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, shortcut_64])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)

        # Block 2
        shortcut_128 = tf.keras.layers.Conv2D(128, (1,1), padding='same')(x) if x.shape[-1] != 128 else x
        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, shortcut_128])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)

        # Block 3
        shortcut_256 = tf.keras.layers.Conv2D(256, (1,1), padding='same')(x) if x.shape[-1] != 256 else x
        x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, shortcut_256])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Dense
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(8, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary(line_length=100)
model.compile(
    loss=config.LOSS,
    optimizer=config.OPTIMIZER,
    metrics=[config.ACCURACY, config.AUC_VALUE, config.PRECISION, config.RECALL],
)

# %%
import gc
gc.collect()

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Approximate class weights by looking at the presence of each class (multi-label)
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(8), y=np.argmax(training_labels, axis=1))
class_weight = dict(enumerate(class_weights))

callbacks = [
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
  tf.keras.callbacks.ModelCheckpoint(config.CHECKPOINT_PATH, monitor='val_auc_value', save_best_only=True, mode='max', verbose=1),
  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
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
    ('binary_accuracy', 'accuracy'),
    ('loss', 'loss'),
    ('auc_value', 'AUC value'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
]
epochs = range(1, len(history.history['loss']) + 1)
for key, label in metrics:
    plt.plot(epochs, history.history[key], 'r', label=f'Training {label}')
    plt.plot(epochs, history.history[f'val_{key}'], 'y', label=f'Validation {label}')
    plt.title(f'Training and validation {label}')
    plt.legend()
    plt.figure()
plt.show()

# %%
test_list = sorted(f for f in os.listdir(config.TESTING_SOURCE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

print(f"{'File Name':<30} {'Predicted Classes':<30} {'Predicted Labels':<20}")
print(f"Total testing images found: {len(test_list)}")

count_normal = 0
count_single_disease = 0
count_multiple_diseases = 0
count_no_disease = 0

for i in range(len(test_list)):
    source = os.path.join(config.TESTING_SOURCE_PATH, test_list[i])
    img = CLAHE(source, config.TARGET_SIZE, 20, (10,10))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    images = np.vstack([img_array])
    predict = model.predict(images)
    predict = predict.reshape(8)
    predicted_labels = (predict >= 0.5).astype(int)

    # Count prediction patterns
    num_pos = np.sum(predicted_labels)
    if num_pos == 0: count_no_disease += 1
    elif num_pos > 1: count_multiple_diseases += 1
    elif predicted_labels[0]: count_normal += 1
    else: count_single_disease += 1

    # Get labels description
    active_labels = [s for s, p in zip(config.LABEL_STRINGS, predicted_labels) if p] or ["None"]

    # Format and display results
    filename = os.path.basename(source)
    predicted_str = str(predicted_labels)
    labels_str = ', '.join(active_labels)

    print(f"{filename:<30} {labels_str:<30} {predicted_str:<20}")

print(f"\nTest Results Summary:")
print(f"Total Tested: {len(test_list)}")
print(f"Normal: {count_normal} | Single Disease: {count_single_disease} | Multiple Diseases: {count_multiple_diseases} | No Disease Detected: {count_no_disease}")
