# %% [markdown]
# # ODIR-5K Multi-Class Classification Pipeline

# %% [markdown]
# ## 1. Setup and Dependencies

# %% [markdown]
# ### 1.1 Import Libraries

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

# %% [markdown]
# ### 1.2 Load Dataset

# %%
FILE_NAME = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
df = pd.read_excel(FILE_NAME)
print(df.head())

# %% [markdown]
# ### 1.3 Extract and Process Diagnostic Keywords

# %%
left_eye_keywords = df['Left-Diagnostic Keywords'].copy()
right_eye_keywords = df['Right-Diagnostic Keywords'].copy()

left_eye_keywords = left_eye_keywords.str.split("，").apply(lambda x: list(set(x)))
right_eye_keywords = right_eye_keywords.str.split("，").apply(lambda x: list(set(x)))

print(left_eye_keywords[2])

# %% [markdown]
# ### 1.4 MultiLabelBinarizer Setup

# %%
mlb = MultiLabelBinarizer()

combined_keywords = pd.concat([left_eye_keywords, right_eye_keywords])
mlb.fit(combined_keywords)

all_diagnosis = list(mlb.classes_)
print("Total different keys diagnosis:", len(all_diagnosis))

# %% [markdown]
# ## 2. Keyword Analysis and Processing

# %% [markdown]
# ### 2.1 Extract Single-Label Keywords

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

key_normal = get_key_diagnosis_single(test_df.columns[7])
key_diabetes = get_key_diagnosis_single(test_df.columns[8])
key_glaucoma = get_key_diagnosis_single(test_df.columns[9])
key_cataract = get_key_diagnosis_single(test_df.columns[10])
key_amd = get_key_diagnosis_single(test_df.columns[11])
key_hypertension = get_key_diagnosis_single(test_df.columns[12])
key_myopia = get_key_diagnosis_single(test_df.columns[13])
key_other_disease = get_key_diagnosis_single(test_df.columns[14])

LABEL_STRINGS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
key_all = [key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease]

for i in range(8):
  print(LABEL_STRINGS[i], len(key_all[i]))

print(key_normal)

# %% [markdown]
# ### 2.2 Remove Duplicate Keywords

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

# %% [markdown]
# ### 2.3 Get All Recognized Keywords

# %%
def get_all_recognized_key(key_all):
    key_all_copy = [list(set(keywords)) for keywords in key_all]
    all_keywords = []
    for keywords in key_all_copy:
      all_keywords.extend(keywords)

    return list(set(all_keywords))

all_key_diagnosis = get_all_recognized_key(key_all)
print("Total unique keywords:", len(all_key_diagnosis))

# %% [markdown]
# ### 2.4 Process Double Diagnosis Rows

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

# %% [markdown]
# ### 2.5 Get Keywords from Multi-Label

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

# %% [markdown]
# ### 2.6 Add Non-Recognized Labels

# %%
# manual listing key
keywords_to_process = [('suspected cataract', 3)]
for keyword, disease_group_index in keywords_to_process:
  if keyword in unrecognized_keywords_list and keyword not in all_key_diagnosis:
    key_all[disease_group_index].append(keyword)
    unrecognized_keywords_list.remove(keyword)

print(key_all[3])

# %%
[print("Not in:", keyword) for keyword in unrecognized_keywords_list if keyword not in all_key_diagnosis]
string = 'central serous chorioretinopathy'
print(string in key_other_disease)

# %% [markdown]
# ## 3. Path Configuration and Data Organization

# %% [markdown]
# ### 3.1 Set Paths

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

# %% [markdown]
# ### 3.2 Load and Split Data

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

# %% [markdown]
# ### 3.3 Organize Images by Diagnosis

# %%
def organize_eye_images_by_diagnosis(file_list, source_path, dest_path):
  "Organize eye images into diagnosis-specific directories based on keywords"
  # Mapping from keywords to label directories and key lists
  label_mapping = [
    (key_normal, 'Normal'),
    (key_diabetes, 'Diabetes'),
    (key_glaucoma, 'Glaucoma'),
    (key_cataract, 'Cataract'),
    (key_amd, 'AMD'),
    (key_hypertension, 'Hypertension'),
    (key_myopia, 'Myopia'),
    (key_other_disease, 'Abnormalities')
  ]

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

# %% [markdown]
# ### 3.4 Verify Data Organization

# %%
from pathlib import Path

training_dir = Path(TRAINING_PATH)
total_files = sum(len(list(subdir.glob('*'))) for subdir in training_dir.iterdir() if subdir.is_dir())
print(total_files)

print(len(os.listdir(TRAINING_SOURCE_PATH)))

# %% [markdown]
# ### 3.5 Image Analysis

# %%
from PIL import Image

cataract_image_list = os.listdir(TRAINING_PATH + 'Cataract')
image_path = TRAINING_PATH + 'Cataract/' + cataract_image_list[2]
im = Image.open(image_path)
width, height = im.size
print(width, height, "from", image_path)

# %%
img = image.load_img(image_path)
plt.imshow(img)
img = image.load_img(image_path, target_size=(int(height/16), int(width/16)), interpolation="lanczos")
plt.imshow(img)

# %% [markdown]
# ## 4. Image Configuration

# %% [markdown]
# ### 4.1 Set Target Size and Color Mode

# %%
TARGET_SIZE = (200, 300) # (int(height/16),int(width/16))
COLOR_MODE = 'rgb' # 'grayscale'
SHAPE_ADD = (3,)  # Default to RGB
if COLOR_MODE == 'grayscale':
  SHAPE_ADD = (1,)
if COLOR_MODE == 'rgb':
  SHAPE_ADD = (3,)

# %% [markdown]
# ## 5. Data Loading and Preprocessing

# %% [markdown]
# ### 5.1 Load Raw Datasets

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

# %% [markdown]
# ### 5.2 Preprocessing and Augmentation Pipeline

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

# %% [markdown]
# ## 6. Model Configuration

# %% [markdown]
# ### 6.1 Training Parameters

# %%
N_EPOCH = 25
INPUT_SHAPE = TARGET_SIZE + SHAPE_ADD
LEARNING_RATE = 0.0001
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
# tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

AUC_VALUE = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', multi_label=True)

PRECISION_SCORE = tf.keras.metrics.Precision(name='precision')
RECALL_SCORE = tf.keras.metrics.Recall(name='recall')

MODEL_PATH = 'Trained_Models/ODIR5K-Multi-Class-bottleneck/'
CHECKPOINT_PATH = MODEL_PATH + 'ODIR5K.keras'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

MODEL_SAVE_WEIGHTS = 'weight'
MODEL_SAVE_NAME_H5 = 'ODIR5K.h5'
MODEL_SAVE_NAME_TF = 'ODIR5K_TF'

USE_TRAINING_MODEL = True

if (os.path.isfile(MODEL_PATH + MODEL_SAVE_NAME_H5) or os.path.exists(MODEL_PATH + MODEL_SAVE_NAME_TF)) and USE_TRAINING_MODEL:
  # if os.path.exists(MODEL_PATH + MODEL_SAVE_NAME_TF):
  #   print("Using tf")
  #   model = tf.keras.models.load_model(MODEL_PATH + MODEL_SAVE_NAME_TF)
  if os.path.isfile(MODEL_PATH + MODEL_SAVE_NAME_H5):
    print("Using h5")
    model = tf.keras.models.load_model(MODEL_PATH + MODEL_SAVE_NAME_H5)
  # model.summary()
  output = model.output
else:
  print("No using saved model")

  def create_conv_block(filters, kernel_size=(3,3), activation='relu'):
    return [
      tf.keras.layers.Conv2D(filters, kernel_size, activation=activation),
      tf.keras.layers.Conv2D(filters, kernel_size, activation=activation),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.BatchNormalization()
    ]

  # Build model using convolutional blocks
  model = tf.keras.models.Sequential([
      tf.keras.Input(shape=INPUT_SHAPE),
  ])

  # Add convolutional blocks with increasing filter sizes
  for filters in [32, 64, 128, 256]:
    model.add(create_conv_block(filters))

  # Add dense layers
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.BatchNormalization())
  # tf.keras.layers.Dropout(0.2),
  model.add(tf.keras.layers.Dense(8, activation='softmax'))

model.summary(line_length=100)
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy', PRECISION_SCORE, RECALL_SCORE, AUC_VALUE])

# %% [markdown]
# ### 6.2 Callback Configuration

# %%
checkpoint_path = "Trained_Models/ODIR5K-Multi-Class/ODIR5K.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1)

STOP_ACCURACY = 0.900

class CallbackStop(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > STOP_ACCURACY):
      print("Reached", STOP_ACCURACY * 100, "accuracy so cancelling training!")
      self.model.stop_training = True

callback_stop = CallbackStop()

# %% [markdown]
# ## 7. Model Training

# %% [markdown]
# ### 7.1 Train the Model

# %%
history = model.fit(train_generator, validation_data=validation_generator,
                    epochs=150,
                    steps_per_epoch=50,
                    # batch_size=train_generator.batch_size,
                    # steps_per_epoch = train_generator.samples // train_generator.batch_size,
                    # validation_steps = validation_generator.samples // validation_generator.batch_size,
                    verbose=1,
                    callbacks=[callback_stop]) # cp_callback

# %% [markdown]
# ## 8. Model Saving

# %% [markdown]
# ### 8.1 Save Trained Model

# %%
MODEL_PATH = 'Trained_Models/ODIR5K-Multi-Class-bottleneck/'
CHECKPOINT_PATH = MODEL_PATH + 'ODIR5K.ckpt'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

MODEL_SAVE_WEIGHTS = 'weight'
MODEL_SAVE_NAME_H5 = 'ODIR5K.h5'
MODEL_SAVE_NAME_TF = 'ODIR5K_TF'
model.save_weights(MODEL_PATH)
model.save_weights(MODEL_PATH + MODEL_SAVE_WEIGHTS)
model.save(MODEL_PATH)
model.save(MODEL_PATH + MODEL_SAVE_NAME_H5)
model.save(MODEL_PATH + MODEL_SAVE_NAME_TF, save_format='tf')

# %%
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)
tflite_model = converter.convert()
open(MODEL_PATH + "ODIR5K.tflite", "wb").write(tflite_model)

# %% [markdown]
# ## 9. Model Evaluation

# %% [markdown]
# ### 9.1 Plot Training Results

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

precision = history.history['precision']
val_precision = history.history['val_precision']

recall = history.history['recall']
val_recall = history.history['val_recall']

auc = history.history['auc']
val_auc = history.history['val_auc']

epochs_training = range(1, len(acc)+1)

plt.plot(epochs_training, acc, 'r', label='Training accuracy')
plt.plot(epochs_training, val_acc, 'y', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs_training, loss, 'r', label='Training loss')
plt.plot(epochs_training, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=1)
plt.figure()

plt.plot(epochs_training, precision, 'r', label='Training Precision')
plt.plot(epochs_training, val_precision, 'y', label='Validation Precision')
plt.title('Training and validation Precision')
plt.legend(loc=2)
plt.figure()

plt.plot(epochs_training, auc, 'r', label='Training AUC value')
plt.plot(epochs_training, val_auc, 'y', label='Validation AUC value')
plt.title('Training and validation AUC value')
plt.legend(loc=3)
plt.figure()

plt.plot(epochs_training, recall, 'r', label='Training Recall')
plt.plot(epochs_training, val_recall, 'y', label='Validation Recall')
plt.title('Training and validation Recall')
plt.legend(loc=4)
plt.figure()

plt.show()

# %% [markdown]
# ### 9.2 Label Decoding Function

# %%
def get_key_indices(val):
  class_names = raw_train_ds.class_names
  label_keys = class_names
  label_values = list(range(len(class_names)))
  return label_keys[label_values.index(val)]

# %% [markdown]
# ## 10. Model Testing

# %% [markdown]
# ### 10.1 Test on Sample Images

# %%
test_list = os.listdir(TESTING_PATH)
test_list.sort()

for i in range(0, len(test_list), 50):
  img = image.load_img(TESTING_PATH + test_list[i], target_size=TARGET_SIZE)
  img_plot = plt.imshow(img)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  images = np.vstack([img_array])
  classes = model.predict(images, batch_size=8)
  # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  # classes = probability_model.predict(images,batch_size=10)
  print("File:", TESTING_PATH + test_list[i]," | predicted as:", get_key_indices(np.argmax(classes)), "at:", np.argmax(classes))

# %%
test_list = os.listdir(TESTING_PATH)

test_list.sort()

for i in range(0, 10):
  img = image.load_img(TESTING_PATH + test_list[i], target_size=TARGET_SIZE)
  img_plot = plt.imshow(img)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  images = np.vstack([img_array])
  classes = model.predict(images, batch_size=8)
  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  classes = probability_model.predict(images,batch_size=8)
  print("File:", TESTING_PATH + test_list[i], " | predicted as:", get_key_indices(np.argmax(classes)), "at:", np.argmax(classes), classes)

# %% [markdown]
# ### 10.2 Model Evaluation

# %%
model.evaluate(validation_generator)

# %% [markdown]
# ### 10.3 Batch Prediction Testing

# %%
test_list = os.listdir(TESTING_PATH)

for i in test_list:
  img = tf.keras.preprocessing.image.load_img(TESTING_PATH + i, target_size=TARGET_SIZE)
  # img_plot = plt.imshow(img)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  images = np.vstack([img_array])
  classes = model.predict(images, batch_size=8)
  # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  # classes = probability_model.predict(images, batch_size=10)
  print("File:", TESTING_PATH + i, " | predicted as:", get_key_indices(np.argmax(classes)), "at:", np.argmax(classes), " | x", np.argmax((model.predict(images) > 0.05).astype("int32")))
  # print("Predicted as: ", classes)
