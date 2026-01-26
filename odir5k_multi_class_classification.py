# %% [markdown]
# #Set Dependencies

# %%
import os
import shutil
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
import tensorflow.keras.optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# %%
os.chdir('ODIR-5K')

# %%
from pandas import read_excel

file_name = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
df = read_excel(file_name)
print(df.head())

# %%
left_eye_keywords = df['Left-Diagnostic Keywords'].copy()
right_eye_keywords = df['Right-Diagnostic Keywords'].copy()

# %%
left_eye_keywords = left_eye_keywords.str.split("，").apply(lambda x: list(set(x)))
right_eye_keywords = right_eye_keywords.str.split("，").apply(lambda x: list(set(x)))

# %%
print(left_eye_keywords[2])

# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

res = pd.DataFrame(mlb.fit_transform(right_eye_keywords),
                   columns=mlb.classes_,
                   index=right_eye_keywords.index)

all_diagnosis_left = res.columns.to_list()
print(len(all_diagnosis_left))

res = pd.DataFrame(mlb.fit_transform(left_eye_keywords),
                   columns=mlb.classes_,
                   index=left_eye_keywords.index)

all_diagnosis_right = res.columns.to_list()
print(len(all_diagnosis_right))

all_diagnosis=list(set(all_diagnosis_left+all_diagnosis_right))
print("Total different keys diagnosis:", len(all_diagnosis))

# %%
test_df = df.copy()
double_diagnosis_row = []

def get_key_diagnosis_single(col_name):
	key_diagnosis = []
	global double_diagnosis_row
	store = True
	for row in range(len(test_df[col_name])):
		store = True
		if test_df[col_name][row] == 1:
			for lable in test_df.columns[7:]:
				if lable == col_name:
					continue
				if test_df[lable][row] == 1:
					double_diagnosis_row.append(row)
					store = False
					break

			if store == True:
				for i in right_eye_keywords[row]:
					key_diagnosis.append(i)
				for i in left_eye_keywords[row]:
					key_diagnosis.append(i)

	key_diagnosis = list(set(key_diagnosis))
	return key_diagnosis

key_normal = get_key_diagnosis_single(test_df.columns[7])
key_diabetes = get_key_diagnosis_single(test_df.columns[8])
key_glaucoma = get_key_diagnosis_single(test_df.columns[9])
key_cataract = get_key_diagnosis_single(test_df.columns[10])
key_amd = get_key_diagnosis_single(test_df.columns[11])
key_hypertension = get_key_diagnosis_single(test_df.columns[12])
key_myopia = get_key_diagnosis_single(test_df.columns[13])
key_other_disease = get_key_diagnosis_single(test_df.columns[14])

label_string = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
key_all = [key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease]

for i in range(8):
  print(label_string[i], len(key_all[i]))

print(key_normal)

# %%
print("Intersect by normal:")
for i in range(1,len(key_all)):
  key_all[i] = list(set(key_all[i]) - set(key_all[0]))

for i in range(8):
  print(label_string[i], len(key_all[i]))

print("\nIntersect by other:")
for i in range(len(key_all)):
  for j in range(i,len(key_all)):
    if i == j:
      continue
    else:
      key_all[i] = list(set(key_all[i]) - set(key_all[j]))

for i in range(8):
  print(label_string[i], len(key_all[i]))

# %%
def get_all_recognized_key(m_key_all):
  mall_key_diagnosis = []
  for i in range(len(m_key_all)):
    m_key_all[i] = list(set(m_key_all[i]))
    mall_key_diagnosis = mall_key_diagnosis + list(set(m_key_all[i]))
  return mall_key_diagnosis

# %%
key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease = key_all[0], key_all[1], key_all[2], key_all[3], key_all[4], key_all[5], key_all[6], key_all[7]

all_key_diagnosis = get_all_recognized_key(key_all)
print(len(all_key_diagnosis))

# %%
double_diagnosis_row = list(set(double_diagnosis_row))
print("Double lablel row:",len(double_diagnosis_row))
double_diagnosis_row.sort()
# double_diagnosis_row

# %%
not_listed = []
listed = False
for row in double_diagnosis_row:
  # print(row)
  for i_list in left_eye_keywords[row]:
    # print(i_list)
    listed = False
    for j in key_all:
      if i_list in j:
        listed = True
        break
    if listed == False:
      not_listed.append(i_list)

for row in double_diagnosis_row:
  for i_list in right_eye_keywords[row]:
    listed = False
    for j in key_all:
      if i_list in j:
        listed = True
        break
    if listed == False:
      not_listed.append(i_list)

not_listed = list(set(not_listed))
# not_listed
print("Not listed diagnosis key:", len(not_listed))

# %%
def intersect_from_multi_label(m_key_all):
  m_not_recognized_list = []
  mall_key_diagnosis = []
  for i in range(len(m_key_all)):
    m_key_all[i] = list(set(m_key_all[i]))
    mall_key_diagnosis = mall_key_diagnosis + list(set(m_key_all[i]))
  for row in double_diagnosis_row:
    not_listed_list = []
    listed_list = []
    col_index = []
    ind = []
    temp_list = []
    for i_list in left_eye_keywords[row]:
      if i_list not in mall_key_diagnosis:
        temp_list.append(i_list)
    for i_list in right_eye_keywords[row]:
      if i_list not in mall_key_diagnosis:
        temp_list.append(i_list)

    for i in range(7, len(test_df.columns)):
      if test_df[test_df.columns[i]][row] == 1:
        col_index.append(i - 7)
    temp_list = list(set(temp_list))
    is_contain_abnormal = 7 in col_index
    if len(temp_list) > 0:
      ind = col_index
      for i_list in left_eye_keywords[row]:
        if i_list not in temp_list:
          listed_list.append(i_list)
      for i_list in right_eye_keywords[row]:
        if i_list not in temp_list:
          listed_list.append(i_list)

      for i_list in listed_list:
        for i in col_index:
          if i_list in key_normal:
            continue
          if i_list in m_key_all[i]:
            ind.remove(i)

      if len(ind) == 0 and is_contain_abnormal:
        ind.append(7)
      if len(ind) == 1 and len(temp_list) == 1:
        m_key_all[ind[0]] = m_key_all[ind[0]] + temp_list
        m_key_all[ind[0]] = list(set(m_key_all[ind[0]]))
      else:
        print("Not recognize")
        m_not_recognized_list.append(temp_list[0])
        m_not_recognized_list = list(set(m_not_recognized_list))

    mall_key_diagnosis = []
    for i in m_key_all:
      mall_key_diagnosis = mall_key_diagnosis+list(set(i))
  return m_key_all, m_not_recognized_list

# %%
iterate = True
not_recognized_list = []
while iterate :
  temp_all_key_diagnosis = all_key_diagnosis.copy()
  key_all, not_recognized_list = intersect_from_multi_label(key_all)
  all_key_diagnosis = get_all_recognized_key(key_all)
  # print(len(temp_all_key_diagnosis), len(all_key_diagnosis))
  print(not_recognized_list)
  if len(temp_all_key_diagnosis) == len(all_key_diagnosis):
    print(True)
    iterate = False

# %%
all_key_diagnosis = get_all_recognized_key(key_all)

key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease = key_all[0], key_all[1], key_all[2], key_all[3], key_all[4], key_all[5], key_all[6], key_all[7]

for i in range(8):
  print(label_string[i], len(key_all[i]))

print("All regnized key:", len(all_key_diagnosis))
print("Not recognized key:", list(set(all_diagnosis) - set(all_key_diagnosis)))

# %%
#manual listed key
string = 'suspected cataract'
if string in not_recognized_list and string not in all_key_diagnosis:
  key_all[3].append(string)
  not_recognized_list.remove(string)

# %%
print(key_all[3])

# %%
all_key_diagnosis = get_all_recognized_key(key_all)

key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease = key_all[0], key_all[1], key_all[2], key_all[3], key_all[4], key_all[5], key_all[6], key_all[7]

for i in range(len(key_all)):
  print(label_string[i], len(key_all[i]))

print("All regnized key:", len(all_key_diagnosis))
print("Not recognized key:", list(set(all_diagnosis)-set(all_key_diagnosis)))

# %%
string = 'central serous chorioretinopathy'

for i in not_recognized_list:
  if i not in all_key_diagnosis:
    print("Not in:", i)

print(string in key_other_disease)

# %%
train_dir = 'training'
validation_dir = 'validation'
test_dir = 'testing'

training_source_path = 'ODIR-5K_Training_Images/'
testing_source_path = 'ODIR-5K_Testing_Images/'

training_path = 'training/'
validation_path = 'validation/'
testing_path = 'testing/'

if os.path.exists(training_path) or os.path.exists(validation_path) or os.path.exists(testing_path):
  shutil.rmtree(training_path)
  shutil.rmtree(validation_path)
  shutil.rmtree(testing_path)

os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

for i in label_string:
  os.mkdir(train_dir + '/' + i)
  os.mkdir(validation_dir + '/' + i)
  os.mkdir(test_dir+'/'+i)

# %%
testing_source_files = os.listdir(testing_source_path)
print(len(testing_source_files))
training_source_files = os.listdir(training_source_path)
print(len(training_source_files))

# training_files = training_source_files

fraction = 0.1

nvalidation = int(len(training_source_files) * fraction)
ntraining = len(training_source_files) - nvalidation

validation_files = sample(training_source_files, nvalidation)
training_files = sample(training_source_files, ntraining)
testing_files = testing_source_files
print(len(training_files))
print(len(validation_files))
print(len(testing_files))

# %%
temp_df = df['Left-Fundus']
print(len(temp_df))
print(temp_df[12])
temp_df = df['Right-Fundus']
print(right_eye_keywords[5])
print(testing_files[1])

temp_keywords = right_eye_keywords
print(temp_keywords[12])

# %%
not_sorted_files = []
"using continue because there are files have more than one diagnosis keys"
for file_name in training_files:
  nrow = None
  if 'left' in file_name:
    temp_df = df['Left-Fundus']
    temp_keywords = left_eye_keywords
  elif 'right' in file_name:
    temp_df = df['Right-Fundus']
    temp_keywords = right_eye_keywords

  for row in range(len(temp_df)):
    if file_name == temp_df[row]:
      nrow = row
      break

  if nrow == None:
    # print("file not listed in data")
    shutil.copyfile(training_source_path + file_name, training_path + file_name)
    continue

  for i in temp_keywords[nrow]:
    if i in key_normal:
      shutil.copyfile(training_source_path + file_name, training_path + 'Normal/' + file_name)
      continue
    if i in key_diabetes:
      shutil.copyfile(training_source_path + file_name, training_path + 'Diabetes/' + file_name)
      continue
    if i in key_glaucoma:
      shutil.copyfile(training_source_path + file_name, training_path + 'Glaucoma/' + file_name)
      continue
    if i in key_cataract:
      shutil.copyfile(training_source_path + file_name, training_path + 'Cataract/' + file_name)
      continue
    if i in key_amd:
      shutil.copyfile(training_source_path + file_name, training_path + 'AMD/' + file_name)
      continue
    if i in key_hypertension:
      shutil.copyfile(training_source_path + file_name, training_path + 'Hypertension/' + file_name)
      continue
    if i in key_myopia:
      shutil.copyfile(training_source_path + file_name, training_path + 'Myopia/' + file_name)
      continue
    if i in key_other_disease:
      shutil.copyfile(training_source_path + file_name, training_path + 'Abnormalities/' + file_name)
      continue
    # else:
    print("Not in list key:", "| row:", row, "| file name:", file_name, "| key diagnosis:", i)
    not_sorted_files.append(file_name)
    not_sorted_files=list(set(not_sorted_files))
    # break

print(len(os.listdir(training_path + 'AMD')))
print(len(os.listdir(training_path + 'Abnormalities')))
print(len(os.listdir(training_path + 'Normal')))
print(len(os.listdir(training_path + 'Cataract')))

# %%
for file_name in validation_files:
  nrow = None
  if 'left' in file_name:
    temp_df = df['Left-Fundus']
    temp_keywords = left_eye_keywords
  elif 'right' in file_name:
    temp_df = df['Right-Fundus']
    temp_keywords = right_eye_keywords

  for row in range(len(temp_df)):
    if file_name == temp_df[row]:
      nrow = row
      break

  if nrow == None:
    # print("file not listed in data")
    shutil.copyfile(training_source_path+file_name, validation_path+file_name)
    continue

  for i in temp_keywords[nrow]:
    if i in key_normal:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Normal/' + file_name)
      continue
    if i in key_diabetes:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Diabetes/' + file_name)
      continue
    if i in key_glaucoma:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Glaucoma/' + file_name)
      continue
    if i in key_cataract:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Cataract/' + file_name)
      continue
    if i in key_amd:
      shutil.copyfile(training_source_path + file_name, validation_path + 'AMD/' + file_name)
      continue
    if i in key_hypertension:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Hypertension/' + file_name)
      continue
    if i in key_myopia:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Myopia/' + file_name)
      continue
    if i in key_other_disease:
      shutil.copyfile(training_source_path + file_name, validation_path + 'Abnormalities/' + file_name)
      continue
    # break
    print("Not in list key:", "| row:", row, "| file name:", file_name, "| key diagnosis:", i)
    not_sorted_files.append(file_name)
    not_sorted_files=list(set(not_sorted_files))

print(len(os.listdir(validation_path + 'AMD')))
print(len(os.listdir(validation_path + 'Abnormalities')))
print(len(os.listdir(validation_path + 'Normal')))
print(len(os.listdir(validation_path + 'Cataract')))

# %%
for file_name in testing_files:
  nrow = None
  if 'left' in file_name:
    temp_df = df['Left-Fundus']
    temp_keywords = left_eye_keywords
  if 'right' in file_name:
    temp_df = df['Right-Fundus']
    temp_keywords = right_eye_keywords

  for row in range(len(temp_df)):
    if file_name == temp_df[row]:
      nrow = row
      break

  if nrow == None:
    # print("file not listed in data")
    shutil.copyfile(testing_source_path+file_name, testing_path+file_name)
    continue

  for i in temp_keywords[nrow]:
    if i in key_normal:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Normal/' + file_name)
      continue
    if i in key_diabetes:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Diabetes/' + file_name)
      continue
    if i in key_glaucoma:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Glaucoma/' + file_name)
      continue
    if i in key_cataract:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Cataract/' + file_name)
      continue
    if i in key_amd:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'AMD/' + file_name)
      continue
    if i in key_hypertension:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Hypertension/' + file_name)
      continue
    if i in key_myopia:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Myopia/' + file_name)
      continue
    if i in key_other_disease:
      shutil.copyfile(testing_source_path + file_name, testing_path + 'Abnormalities/' + file_name)
      continue
    print("Not in list key:", "| row: ", row, "| file name: ", file_name, "| key diagnosis:", i)
    not_sorted_files.append(file_name)
    not_sorted_files=list(set(not_sorted_files))

print(len(os.listdir(testing_path + 'AMD')))
print(len(os.listdir(testing_path + 'Abnormalities')))
print(len(os.listdir(testing_path + 'Normal')))
print(len(os.listdir(testing_path + 'Cataract')))
print(len(os.listdir(testing_path)))

# %%
print(not_sorted_files)

# %%
dir_list = os.listdir(training_path)
countx = 0
for i in dir_list:
  countx+=len(os.listdir(training_path+i))

print(countx)

print(len(os.listdir(training_source_path)))

# %%
from PIL import Image

cataract_image_list = os.listdir(training_path + 'Cataract')
image_path = training_path + 'Cataract/' + cataract_image_list[2]
im = Image.open(image_path)
width, height = im.size
print(width, height, "from", image_path)

# %%
img = image.load_img(image_path)
plt.imshow(img)
img = image.load_img(image_path, target_size=(int(height/16), int(width/16)), interpolation="lanczos")
plt.imshow(img)

# %%
# target_size = (int(height/16),int(width/16))
target_size = (200, 300)
# mode = 'grayscale'
color_mode = 'rgb'
if color_mode == 'grayscale':
  shape_add = (1,)
if color_mode == 'rgb':
  shape_add = (3,)

# %%
# 1. Load Raw Datasets
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    training_path,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=32,
    image_size=target_size,
    shuffle=True,
    seed=42,
    interpolation='lanczos3'
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=32,
    image_size=target_size,
    shuffle=False,
    interpolation='lanczos3'
)

# Capture class names for label decoding
class_names = raw_train_ds.class_names

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
n_epoch = 25
input_shape = target_size + shape_add
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# tf.keras.optimizers.SGD(learning_rate=learning_rate)

auc_value = tf.keras.metrics.AUC(
                                  # name='AUC values',
                                  num_thresholds=200,
                                  curve='ROC',
                                  summation_method='interpolation',
                                  # threshold=0.5,
                                  multi_label=True)

precision_score = tf.keras.metrics.Precision(name='precision')
recall_score = tf.keras.metrics.Recall(name='recall')

model_path = 'Trained_Models/ODIR5K-bottleneck/'
checkpoint_path = model_path + 'ODIR5K.keras'
checkpoint_dir = os.path.dirname(checkpoint_path)

model_save_weights = 'weight'
model_save_name_h5 = 'ODIR5K.h5'
model_save_name_tf = 'ODIR5K_TF'

use_training_model = True

if (os.path.isfile(model_path + model_save_name_h5) or os.path.exists(model_path + model_save_name_tf)) and use_training_model:
  # if os.path.exists(model_path+model_save_name_tf):
  #   print("Using tf")
  #   model = tf.keras.models.load_model(model_path+model_save_name_tf)
  if os.path.isfile(model_path + model_save_name_h5):
    print("Using h5")
    model = tf.keras.models.load_model(model_path + model_save_name_h5)
  # model.summary()
  output = model.output
else:
  print("No using saved model")
  model = tf.keras.models.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    # The fourth convolution
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='softmax')
  ])
  # model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary(line_length=100)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', precision_score, recall_score, auc_value])

# %%
checkpoint_path = "Trained_Models/ODIR5K/ODIR5K.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                #  save_weights_only=True,
                                                 verbose=1)

stop_accuracy = 0.900

# Define a Callback class that stops training once accuracy reaches the certain accuracy
class CallbackStop(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > stop_accuracy):
      print("Reached", stop_accuracy * 100, "accuracy so cancelling training!")
      self.model.stop_training = True

callback_stop = CallbackStop()

# %%
history = model.fit(train_generator, validation_data=validation_generator,
                    epochs=150,
                    steps_per_epoch=50,
                    # batch_size=train_generator.batch_size,
                    # steps_per_epoch = train_generator.samples // train_generator.batch_size,
                    # validation_steps = validation_generator.samples // validation_generator.batch_size,
                    verbose=1,
                    callbacks=[callback_stop,
                              #  cp_callback
                               ])

# %%
model_path = 'Trained_Models/ODIR5K-bottleneck/'
checkpoint_path = model_path + 'ODIR5K.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

model_save_weights = 'weight'
model_save_name_h5 = 'ODIR5K.h5'
model_save_name_tf = 'ODIR5K_TF'
model.save_weights(model_path)
model.save_weights(model_path + model_save_weights)
model.save(model_path)
model.save(model_path + model_save_name_h5)
model.save(model_path + model_save_name_tf, save_format='tf')

# %%
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()
open(model_path + "ODIR5K.tflite", "wb").write(tflite_model)

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

# %%
# import tensorflow as tf
# import os
# import numpy as np

# model_path = 'Trained_Models/ODIR5K-bottleneck/'
# model_save_name_h5 = 'ODIR5K.h5'
# model = tf.keras.models.load_model(model_path + model_save_name_h5)

# %%
label_keys = class_names
label_values = list(range(len(class_names)))

def get_key_indices(val):
  return label_keys[label_values.index(val)]

# %%
test_list = os.listdir(testing_path)
test_list.sort()

for i_file in range(0, len(test_list), 50):
  img = image.load_img(testing_path + test_list[i_file], target_size=target_size,)
  img_plot = plt.imshow(img)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  images = np.vstack([img_array])
  classes = model.predict(images, batch_size=8)
  # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  # classes = probability_model.predict(images,batch_size=10)
  print("File:", testing_path + test_list[i_file]," | predicted as:", get_key_indices(np.argmax(classes)), "at:", np.argmax(classes))

# %%
test_list = os.listdir(testing_path)

test_list.sort()

for i_file in range(0, 10):
  img = image.load_img(testing_path + test_list[i_file], target_size=target_size)
  img_plot = plt.imshow(img)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  images = np.vstack([img_array])
  classes = model.predict(images, batch_size=8)
  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  classes = probability_model.predict(images,batch_size=8)
  print("File:", testing_path+test_list[i_file]," | predicted as:", get_key_indices(np.argmax(classes)), "at:", np.argmax(classes), classes)

# %%
model.evaluate(validation_generator)

# %%
model.predict('ODIR-5K/testing/1000_left.jpg')

# %%
# for i_file in test_list:
img = tf.keras.preprocessing.image.load_img('ODIR-5K/testing/1604_left.jpg', target_size=target_size)
img_plot = plt.imshow(img)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

images = np.vstack([img_array])
classes = model.predict_classes(images)
print(classes)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
classes = probability_model.predict(images)
print(classes)
print("File:", testing_path + test_list[300])
print("Predicted as:", np.argmax(classes))
# np.argmax((probability_model.predict(images) > 0.5).astype("int32"))

# %%
# for i_file in test_list:
img = tf.keras.preprocessing.image.load_img('ODIR-5K/validation/Abnormalities/1031_left.jpg', target_size=target_size,)
# img = tf.keras.preprocessing.image.load_img('/ODIR-5K/training/Diabetes/1022_left.jpg', target_size=target_size)
img_plot = plt.imshow(img)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

images = np.vstack([img_array])
classes = model.predict(images)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
classes = probability_model.predict(images)
print("File:", testing_path + test_list[300])
print("Predicted as:", np.argmax(classes))

# %%
predict_fun = model.make_predict_function

model.predict(images)

# %%
test_list = os.listdir(testing_path)

for i_file in test_list:
  img = tf.keras.preprocessing.image.load_img(testing_path + i_file, target_size=target_size)
  # img_plot = plt.imshow(img)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  images = np.vstack([img_array])
  classes = model.predict(images, batch_size=8)
  # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  # classes = probability_model.predict(images,batch_size=10)
  print("File:", testing_path + i_file," | predicted as:", get_key_indices(np.argmax(classes)), "at:", np.argmax(classes), " | x", np.argmax((model.predict(images) > 0.05).astype("int32")))
  # print("predicted as : ", classes)

# %%
test_dir = 'ODIR-5K/testing/'

print(len(os.listdir(testing_path)))

# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(test_dir, target_size=target_size, color_mode="rgb", shuffle=False, class_mode='categorical', batch_size=1)

# filenames = test_generator.filenames
# nb_samples = len(filenames)

# predict = model.predict_generator(test_generator)
# np.argmax(predict[0])
# for i in range (500):
#   print(np.argmax(predict[i]))
