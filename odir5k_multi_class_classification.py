# %%
import os
import shutil
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
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
for i in range(1, len(key_all_sets)):
  key_all_sets[i] -= normal_keywords

# Remove duplicate keywords between groups
for i in range(len(key_all_sets)):
  for j in range(i+1, len(key_all_sets)):
    # Find intersection and remove from the group with higher index
    intersection = key_all_sets[i] & key_all_sets[j]
    key_all_sets[j] -= intersection

# Convert back to list
for i in range(len(key_all_sets)):
  key_all[i] = list(key_all_sets[i])

# Print results
print("Intersect by normal:")
for i in range(len(key_all)):
  print(LABEL_STRINGS[i], len(key_all[i]))

print("Intersect by other:")
for i in range(len(key_all)):
  print(LABEL_STRINGS[i], len(key_all[i]))

# %%
def get_all_recognized_key(key_all):
    key_all_copy = [list(set(keywords)) for keywords in key_all]
    all_keywords = []
    for keywords in key_all_copy:
      all_keywords.extend(keywords)

    return list(set(all_keywords))

all_key_diagnosis = get_all_recognized_key(key_all)
print("Total unique keywords:", len(all_key_diagnosis))

# %%
double_diagnosis_row = list(set(double_diagnosis_row))
print("Double label row:", len(double_diagnosis_row))
double_diagnosis_row.sort()

# %%
all_known_keywords = set()
for keywords in key_all:
  all_known_keywords.update(keywords)
not_listed = set()

for row in double_diagnosis_row:
  for keyword in left_eye_keywords[row]:
    if keyword not in all_known_keywords:
      not_listed.add(keyword)

  for keyword in right_eye_keywords[row]:
    if keyword not in all_known_keywords:
      not_listed.add(keyword)

not_listed = list(not_listed)
print("Not listed diagnosis key:", len(not_listed))

# %%
def intersect_from_multi_label(keyword_groups):
  known_keywords = set()
  for disease_keywords in keyword_groups:
    known_keywords.update(disease_keywords)
  unrecognized_keywords = set()

  for record_index in double_diagnosis_row:
    undiscovered_keywords = set()
    for keyword in left_eye_keywords[record_index] + right_eye_keywords[record_index]:
      if keyword not in known_keywords:
        undiscovered_keywords.add(keyword)

    if undiscovered_keywords:
      related_disease_groups = []
      for column_index in range(7, len(test_df.columns)):
        if test_df[test_df.columns[column_index]][record_index] == 1:
          related_disease_groups.append(column_index - 7)

      if len(related_disease_groups) == 1 and len(undiscovered_keywords) == 1:
        disease_group_index = related_disease_groups[0]
        new_keyword = undiscovered_keywords.pop()
        keyword_groups[disease_group_index].append(new_keyword)
        known_keywords.add(new_keyword)
      else:
          unrecognized_keywords.update(undiscovered_keywords)

  return keyword_groups, list(unrecognized_keywords)

processing_required = True
unrecognized_keywords_list = []

while processing_required:
  previous_keyword_count = len(all_key_diagnosis)
  key_all, unrecognized_keywords_list = intersect_from_multi_label(key_all)
  all_key_diagnosis = get_all_recognized_key(key_all)
  print(unrecognized_keywords_list)
  current_keyword_count = len(all_key_diagnosis)
  if current_keyword_count == previous_keyword_count:
    print(True)
    processing_required = False

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
print(len(testing_source_files))
training_source_files = os.listdir(TRAINING_SOURCE_PATH)
print(len(training_source_files))

VALIDATION_FRACTION = 0.1
N_VALIDATION = int(len(training_source_files) * VALIDATION_FRACTION)
N_TRAINING = len(training_source_files) - N_VALIDATION

validation_files = sample(training_source_files, N_VALIDATION)
training_files = sample(training_source_files, N_TRAINING)
testing_files = testing_source_files
print(len(training_files))
print(len(validation_files))
print(len(testing_files))

# %%
def organize_eye_images_by_diagnosis(file_list, source_path, dest_path):
  "Organize eye images into diagnosis-specific directories based on keywords"
  label_mapping = list(zip(key_all, LABEL_STRINGS))

  for file_name in file_list:
    nrow = None
    if 'left' in file_name:
      tmp_df = df['Left-Fundus']
      tmp_keywords = left_eye_keywords
    elif 'right' in file_name:
      tmp_df = df['Right-Fundus']
      tmp_keywords = right_eye_keywords

    for row in range(len(tmp_df)):
      if file_name == tmp_df[row]:
        nrow = row
        break

    if nrow is None:
      shutil.copyfile(source_path + file_name, dest_path + file_name)
      continue

    for key_list, label_dir in label_mapping:
      if any(keyword in key_list for keyword in tmp_keywords[nrow]):
        shutil.copyfile(source_path + file_name, dest_path + label_dir + '/' + file_name)
        break

# Process training files
organize_eye_images_by_diagnosis(training_files, TRAINING_SOURCE_PATH, TRAINING_PATH)

print(len(os.listdir(TRAINING_PATH + 'AMD')))
print(len(os.listdir(TRAINING_PATH + 'Abnormalities')))
print(len(os.listdir(TRAINING_PATH + 'Normal')))
print(len(os.listdir(TRAINING_PATH + 'Cataract')))

# Process validation files
organize_eye_images_by_diagnosis(validation_files, TRAINING_SOURCE_PATH, VALIDATION_PATH)

print(len(os.listdir(VALIDATION_PATH + 'AMD')))
print(len(os.listdir(VALIDATION_PATH + 'Abnormalities')))
print(len(os.listdir(VALIDATION_PATH + 'Normal')))
print(len(os.listdir(VALIDATION_PATH + 'Cataract')))

# Process testing files
organize_eye_images_by_diagnosis(testing_files, TESTING_SOURCE_PATH, TESTING_PATH)

print(len(os.listdir(TESTING_PATH + 'AMD')))
print(len(os.listdir(TESTING_PATH + 'Abnormalities')))
print(len(os.listdir(TESTING_PATH + 'Normal')))
print(len(os.listdir(TESTING_PATH + 'Cataract')))
print(len(os.listdir(TESTING_PATH)))

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
TARGET_SIZE = (200, 300) # (int(height/16),int(width/16))
COLOR_MODE = 'rgb' # 'grayscale'
SHAPE_ADD = (3,)  # Default to RGB
if COLOR_MODE == 'grayscale':
  SHAPE_ADD = (1,)
if COLOR_MODE == 'rgb':
  SHAPE_ADD = (3,)

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

# %%
# 2. Define Preprocessing/Augmentation Pipeline
augmentation_layers = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(factor=40/360, fill_mode='nearest'), # rotation_range=40
  tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode='nearest'), # zoom_range=0.2
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

STOP_ACCURACY = 0.90

class CallbackStop(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > STOP_ACCURACY):
      print("Reached", STOP_ACCURACY * 100, "accuracy so cancelling training!")
      self.model.stop_training = True

callback_stop = CallbackStop()
callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, verbose=1)

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
                    steps_per_epoch=50,
                    # batch_size=train_generator.batch_size,
                    # steps_per_epoch = train_generator.samples // train_generator.batch_size,
                    # validation_steps = validation_generator.samples // validation_generator.batch_size,
                    verbose=1,
                    callbacks=[callback_stop]) # callback_cp

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

def get_key_indices(val):
  class_names = raw_train_ds.class_names
  label_keys = class_names
  label_values = list(range(len(class_names)))
  return label_keys[label_values.index(val)]

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
    pred_label = get_key_indices(pred_idx)
    x_idx = np.argmax((classes > 0.05).astype("int32"))

    print(f"File: {img_path} | predicted as: {pred_label} at: {pred_idx} | x: {x_idx} | probs: {prob_classes}")
