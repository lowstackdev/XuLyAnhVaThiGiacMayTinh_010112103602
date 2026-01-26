# %%
import os
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
os.chdir('ODIR-5K')

# %%
FILE_NAME = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
df = pd.read_excel(FILE_NAME)
print(df.head())

# %%
left_eye_keywords = df['Left-Diagnostic Keywords'].copy()
right_eye_keywords = df['Right-Diagnostic Keywords'].copy()

left_eye_keywords = left_eye_keywords.str.split("，").apply(lambda x: list(set(x)))
right_eye_keywords = right_eye_keywords.str.split("，").apply(lambda x: list(set(x)))

print(left_eye_keywords[2])

# %%
mlb = MultiLabelBinarizer()

combined_keywords = pd.concat([left_eye_keywords, right_eye_keywords])
mlb.fit(combined_keywords)

all_diagnosis = list(mlb.classes_)
print("Total different keys diagnosis:", len(all_diagnosis))

# %%
test_df = df.copy()

# Compute double_diagnosis_row once (rows with multiple diagnoses)
diag_cols = test_df.columns[7:]
double_diagnosis_row = test_df[test_df[diag_cols].sum(axis=1) > 1].index.tolist()

def get_key_diagnosis_single(col_name):
  # Get other diagnosis columns
  other_diag_cols = [col for col in diag_cols if col != col_name]

  # Find rows where target column == 1 AND all other diagnosis columns == 0
  single_rows = test_df[(test_df[col_name] == 1) & (test_df[other_diag_cols].sum(axis=1) == 0)].index

  # Collect unique keywords from left and right eye for these rows
  key_diagnosis = []
  for row in single_rows:
    key_diagnosis.extend(left_eye_keywords[row])
    key_diagnosis.extend(right_eye_keywords[row])

  return list(set(key_diagnosis))

LABEL_STRINGS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
key_all = [get_key_diagnosis_single(test_df.columns[7 + i]) for i in range(8)]

for i in range(8):
  print(LABEL_STRINGS[i], len(key_all[i]))

# %%
key_all_sets = [set(keywords) for keywords in key_all]

# Remove "normal" keyword from all groups
normal_keywords = key_all_sets[0]
key_all_sets[1:] = [keywords - normal_keywords for keywords in key_all_sets[1:]]

# Remove duplicate keywords between groups
for i, keywords_i in enumerate(key_all_sets):
  for keywords_j in key_all_sets[i+1:]:
    keywords_j -= keywords_i & keywords_j

# Convert back to list
key_all[:] = [list(keywords) for keywords in key_all_sets]

# Print results
print("Intersected:")
for i in range(len(key_all)):
  print(LABEL_STRINGS[i], len(key_all[i]))

# %%
def get_all_recognized_key(key_all):
  return list(set([keyword for keywords in key_all for keyword in set(keywords)]))

all_key_diagnosis = get_all_recognized_key(key_all)
print("Total unique keywords:", len(all_key_diagnosis))

# %%
double_diagnosis_row = sorted(set(double_diagnosis_row))
print("Double label row:", len(double_diagnosis_row))

# %%
all_known_keywords = set().union(*key_all)
not_listed = {keyword for row in double_diagnosis_row
              for keyword in left_eye_keywords[row] + right_eye_keywords[row]
              if keyword not in all_known_keywords}
print("Not listed diagnosis key:", len(not_listed))

# %%
def intersect_from_multi_label(keyword_groups):
  known_keywords = set().union(*keyword_groups)
  unrecognized_keywords = set()

  for record_idx in double_diagnosis_row:
    keywords = left_eye_keywords[record_idx] + right_eye_keywords[record_idx]
    undiscovered = set(kw for kw in keywords if kw not in known_keywords)

    if undiscovered:
      related_groups = [col_idx - 7 for col_idx in range(7, len(test_df.columns))
                        if test_df.iloc[record_idx, col_idx] == 1]

      if len(related_groups) == 1 and len(undiscovered) == 1:
        keyword_groups[related_groups[0]].append(undiscovered.pop())
        known_keywords.add(keyword_groups[related_groups[0]][-1])
      else:
        unrecognized_keywords.update(undiscovered)

  return keyword_groups, list(unrecognized_keywords)

# Process until convergence
prev_count = 0
while True:
  prev_count = len(all_key_diagnosis)
  key_all, unrecognized_keywords_list = intersect_from_multi_label(key_all)
  all_key_diagnosis = get_all_recognized_key(key_all)
  print(unrecognized_keywords_list)
  if len(all_key_diagnosis) == prev_count:
    print(True)
    break

# %%
TRAINING_SOURCE_PATH = 'ODIR-5K_Training_Images/'
TESTING_SOURCE_PATH = 'ODIR-5K_Testing_Images/'

TRAINING_PATH = 'training/'
VALIDATION_PATH = 'validation/'
TESTING_PATH = 'testing/'

# Remove existing directories if they exist
for path in [TRAINING_PATH, VALIDATION_PATH, TESTING_PATH]:
  if os.path.exists(path):
    shutil.rmtree(path)

# Create main directories and subdirectories for each label
for path in [TRAINING_PATH, VALIDATION_PATH, TESTING_PATH]:
  for label in LABEL_STRINGS:
    os.makedirs(os.path.join(path, label), exist_ok=True)

# %%
testing_source_files = os.listdir(TESTING_SOURCE_PATH)
print(f"Total testing source images: {len(testing_source_files)}")

training_source_files = os.listdir(TRAINING_SOURCE_PATH)
print(f"Total training source images: {len(training_source_files)}")

VALIDATION_FRACTION = 0.1

# Group files by patient ID to prevent data leakage (same patient's eyes in different sets)
# File naming convention: [PatientID]_[eye].jpg
from collections import defaultdict
patient_to_files = defaultdict(list)
for f in training_source_files:
  patient_id = f.split('_')[0]
  patient_to_files[patient_id].append(f)

unique_patient_ids = list(patient_to_files.keys())
n_val_patients = int(len(unique_patient_ids) * VALIDATION_FRACTION)

validation_patient_ids = sample(unique_patient_ids, n_val_patients)
training_patient_ids = [pid for pid in unique_patient_ids if pid not in validation_patient_ids]

validation_files = [f for pid in validation_patient_ids for f in patient_to_files[pid]]
training_files = [f for pid in training_patient_ids for f in patient_to_files[pid]]
testing_files = testing_source_files
print(f"Total validation files: {len(validation_files)}")
print(f"Total training files: {len(training_files)}")
print(f"Total testing files: {len(testing_files)}")

# %%
def organize_eye_images_by_diagnosis(file_list, source_path, dest_path):
  "Organize eye images into diagnosis-specific directories based on keywords"
  label_mapping = list(zip(key_all, LABEL_STRINGS))

  EYE_DATA = [
    ('Left-Fundus', left_eye_keywords),
    ('Right-Fundus', right_eye_keywords)
  ]

  for file_name in file_list:
    # Handle testing files with different naming convention (e.g., "1000_left.jpg")
    if '_left' in file_name or '_right' in file_name:
      # Extract base filename without _left/_right suffix for matching
      base_name = file_name.replace('_left', '').replace('_right', '').replace('.jpg', '')
      matching_files = [f for f in df['Left-Fundus'] if base_name in f] + [f for f in df['Right-Fundus'] if base_name in f]

      if matching_files:
        nrow, keywords_data = next(
            ((i, keywords) for col, keywords in EYE_DATA
             for i, val in enumerate(df[col]) if base_name in val),
            (None, None)
        )
      else:
        nrow, keywords_data = None, None
    else:
      nrow, keywords_data = next(
          ((i, keywords) for col, keywords in EYE_DATA
           for i, val in enumerate(df[col]) if val == file_name),
          (None, None)
      )

    if nrow is None:
      # If no match found, copy to the first category (Normal) as default
      shutil.copy(source_path + file_name, os.path.join(dest_path, LABEL_STRINGS[0]))
      continue

    for key_list, label_dir in label_mapping:
      if any(keyword in key_list for keyword in keywords_data[nrow]):
        shutil.copy(source_path + file_name, os.path.join(dest_path, label_dir))
        break

for files, src, dest, name in [
  (training_files, TRAINING_SOURCE_PATH, TRAINING_PATH, "Training"),
  (validation_files, TRAINING_SOURCE_PATH, VALIDATION_PATH, "Validation"),
  (testing_files, TESTING_SOURCE_PATH, TESTING_PATH, "Testing")
]:
  print(f"\nOrganizing {name} files...")
  organize_eye_images_by_diagnosis(files, src, dest)
  for label in LABEL_STRINGS:
    count = len(os.listdir(os.path.join(dest, label)))
    print(f"{name} {label} count: {count}")

# %%
# from pathlib import Path

# training_dir = Path(TRAINING_PATH)
# total_files = sum(len(list(subdir.glob('*'))) for subdir in training_dir.iterdir() if subdir.is_dir())
# print(total_files)

# print(len(os.listdir(TRAINING_SOURCE_PATH)))

# %%
# from PIL import Image

# cataract_image_list = os.listdir(TRAINING_PATH + 'Cataract')
# image_path = TRAINING_PATH + 'Cataract/' + cataract_image_list[2]
# im = Image.open(image_path)
# width, height = im.size
# print(width, height, "from", image_path)

# %%
# img = image.load_img(image_path)
# plt.imshow(img)
# img = image.load_img(image_path, target_size=(int(height/16), int(width/16)), interpolation="lanczos")
# plt.imshow(img)

# %%
TARGET_SIZE = (200, 300) # (int(height/16), int(width/16))
COLOR_MODE = 'rgb'
COLOR_SHAPE_MAP = {
  'grayscale': (1,),
  'rgb': (3,),
  'rgba': (4,)
}
SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

# %%
# 1. Load Raw Datasets
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
  TRAINING_PATH,
  labels='inferred',
  label_mode='categorical',
  color_mode=COLOR_MODE,
  batch_size=32,
  image_size=TARGET_SIZE,
  shuffle=True,
  seed=42,
  interpolation='lanczos3'
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
  VALIDATION_PATH,
  labels='inferred',
  label_mode='categorical',
  color_mode=COLOR_MODE,
  batch_size=32,
  image_size=TARGET_SIZE,
  shuffle=False,
  interpolation='lanczos3'
)

# 2. Define Preprocessing/Augmentation Pipeline
augmentation_layers = tf.keras.Sequential([
  # 1. GEOMETRIC TRANSFORMATIONS (limited)
  tf.keras.layers.RandomRotation(
      factor=0.1,  # ±36 degrees - small to preserve orientation
      fill_mode='nearest'
  ),
  tf.keras.layers.RandomZoom(
      height_factor=0.15,
      width_factor=0.15,  # Uniform zoom in both dimensions
      fill_mode='nearest'
  ),
  tf.keras.layers.RandomTranslation(
      height_factor=0.05,  # Only 5% - very small
      width_factor=0.05,
      fill_mode='nearest'
  ),

  # 2. PHOTOMETRIC TRANSFORMATIONS (important)
  tf.keras.layers.RandomBrightness(
      max_delta=0.15,  # 15% - not too large
      value_range=(0, 1)  # Pixel values are already normalized
  ),
  tf.keras.layers.RandomContrast(
      factor=0.15  # 15% contrast variation
  ),

  # 3. NOISE & ARTIFACTS (real-world simulation)
  tf.keras.layers.GaussianNoise(
      stddev=0.01  # Small noise
  ),

  # 4. BLUR (simulating focus issues)
  tf.keras.layers.RandomZoom(
      height_factor=(-0.02, 0.02),  # Minor blur effect
      width_factor=(-0.02, 0.02),
      fill_mode='nearest'
  ),
])

rescaling_layer = tf.keras.layers.Rescaling(1./255)

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

class_weights = compute_class_weight(
  'balanced',
  classes=np.unique(train_labels),
  y=train_labels
)

# %%
USE_MODEL = "using custom"
USE_PRETRAINED_MODEL = True
INPUT_SHAPE = TARGET_SIZE + SHAPE_ADD

N_EPOCH = 150
LEARNING_RATE = 0.0001
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE) # tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

MODEL_PATH = 'Trained_Models/ODIR5K-Multi-Class/'
MODEL_SAVE_WEIGHTS = 'weight.h5'
MODEL_SAVE_NAME_H5 = 'ODIR5K.h5'

CHECKPOINT_PATH = MODEL_PATH + 'ODIR5K.keras'

AUC_VALUE = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', multi_label=True)
PRECISION_SCORE = tf.keras.metrics.Precision(name='precision')
RECALL_SCORE = tf.keras.metrics.Recall(name='recall')

callbacks = [
  tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
  ),
  tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
  ),
  tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
  )
]

# %%
if os.path.isfile(MODEL_PATH + MODEL_SAVE_NAME_H5) and USE_PRETRAINED_MODEL:
  print("Using h5")
  model = tf.keras.models.load_model(MODEL_PATH + MODEL_SAVE_NAME_H5)
else:
  print("No using saved model")
  if USE_MODEL == "using custom":
    def create_conv_block(filters, kernel_size=(3,3), activation='relu'):
      return [
        tf.keras.layers.Conv2D(filters, kernel_size, activation=activation),
        tf.keras.layers.Conv2D(filters, kernel_size, activation=activation),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.BatchNormalization()
      ]

    model = tf.keras.models.Sequential([
      tf.keras.Input(shape=INPUT_SHAPE),
      *create_conv_block(32),
      *create_conv_block(64),
      *create_conv_block(128),
      *create_conv_block(256),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(8, activation='softmax')
    ])

model.summary(line_length=100)
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy', PRECISION_SCORE, RECALL_SCORE, AUC_VALUE])

# %%
history = model.fit(train_generator, validation_data=validation_generator,
                    epochs=N_EPOCH,
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    validation_steps=validation_generator.samples//validation_generator.batch_size,
                    class_weights=class_weights,
                    verbose=1,
                    callbacks=callbacks)

# %%
model.save_weights(MODEL_PATH + MODEL_SAVE_WEIGHTS)
model.save(MODEL_PATH + MODEL_SAVE_NAME_H5)

# %%
metrics = [
  ('accuracy', 'accuracy', 0),
  ('loss', 'loss', 1),
  ('precision', 'Precision', 2),
  ('auc', 'AUC value', 3),
  ('recall', 'Recall', 4)
]
epochs = range(1, len(history.history['accuracy']) + 1)
for key, label, loc in metrics:
    plt.plot(epochs, history.history[key], 'r', label=f'Training {label}')
    plt.plot(epochs, history.history[f'val_{key}'], 'y', label=f'Validation {label}')
    plt.title(f'Training and validation {label}')
    plt.legend(loc=loc)
    plt.figure()
plt.show()

# %%
model.evaluate(validation_generator)

test_list = sorted(os.listdir(TESTING_PATH))
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

print(f"\nPredicting {len(test_list)} files in {TESTING_PATH}...")
for file_name in test_list:
  img_path = TESTING_PATH + file_name
  img = image.load_img(img_path, target_size=TARGET_SIZE)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  classes = model.predict(img_array, batch_size=8, verbose=0)
  prob_classes = probability_model.predict(img_array, batch_size=8, verbose=0)

  pred_idx = np.argmax(classes)
  pred_label = raw_train_ds.class_names[pred_idx]
  x_idx = np.argmax((classes > 0.05).astype("int32"))

  print(f"File: {img_path} | predicted as: {pred_label} at: {pred_idx} | x: {x_idx} | probs: {prob_classes}")
