# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import tensorflow as tf

print(tf.__version__)

# %%
os.chdir('ODIR-5K')

# %%
from pandas import read_excel

FILE_NAME = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
df = read_excel(FILE_NAME)
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
key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease = key_all

for i in range(8):
    print(LABEL_STRINGS[i], len(key_all[i]))

print(key_normal)

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
print("Double label row", len(double_diagnosis_row))
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
                    related_disease_groups.append(column_index-7)

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
# Define method for image resize, cropping and image Contrast Limited Adaptive Histogram Equalization (CLAHE)
# using Opencv 4

def image_resize(image_path, dim):
    img = cv2.imread(image_path)
    if img.shape[1] != img.shape[0]:
        x = img.shape[1] // 2
        y = img.shape[0] // 2
        x = x-y
        img = img[0:0 + img.shape[0], x:x + img.shape[0]]
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def CLAHE(image_path, dim, clipLimit, tileGridSize):
    img = image_resize(image_path, dim)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# %%
# Set target size image
TARGET_SIZE = (230, 230)
# COLOR_MODE = 'grayscale'
COLOR_MODE = 'rgb'
SHAPE_ADD = (3,)  # Default to RGB
if COLOR_MODE == 'grayscale':
	SHAPE_ADD = (1,)
elif COLOR_MODE == 'rgb':
	SHAPE_ADD = (3,)

# %%
# Function for generate label to single image

# Return index in key of all diagnosis list
def get_index_label(key, key_all):
    for i in key_all:
        if key in i:
            return key_all.index(i)

# Return multilabel by index
def get_multi_label_from_keys(idx_label):
    tmp_label = []
    for i in range(8):
        if i in idx_label:
            tmp_label.append(1)
        else:
            tmp_label.append(0)
    return tmp_label

# %%
import concurrent.futures
from functools import partial

def process_fundus_image_with_clahe(img_path, keywords, key_all, target_size):
    """Process a single fundus image with CLAHE enhancement and generate diagnostic labels"""
    try:
        # Read image
        fundus_img = cv2.imread(img_path)
        if fundus_img is None:
            return None, None, None

        # Process keywords to generate multi-label diagnosis
        indices = [get_index_label(key, key_all) for key in keywords]
        indices = list(set(indices))
        label = get_multi_label_from_keys(indices)

        # Process image with CLAHE enhancement
        clahe_img = CLAHE(img_path, target_size, 20, (10,10))

        return label, os.path.basename(img_path), clahe_img

    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None, None, None

def process_patient_record_parallel(row_idx, df, left_eye_keywords, right_eye_keywords, key_all, target_size):
    """Process a single patient record (both eyes) in parallel and generate diagnostic data"""
    results = []
    try:
        # Process left eye fundus image
        left_img_path = os.path.join('ODIR-5K_Training_Images', df['Left-Fundus'][row_idx])
        left_label, left_feature, left_clahe = process_fundus_image_with_clahe(left_img_path, left_eye_keywords[row_idx], key_all, target_size)
        if left_label is not None:
            results.append((left_label, left_feature, left_clahe))

        # Process right eye fundus image
        right_img_path = os.path.join('ODIR-5K_Training_Images', df['Right-Fundus'][row_idx])
        right_label, right_feature, right_clahe = process_fundus_image_with_clahe(right_img_path, right_eye_keywords[row_idx], key_all, target_size)
        if right_label is not None:
            results.append((right_label, right_feature, right_clahe))

    except Exception as e:
        print(f"Error processing patient record {row_idx}: {str(e)}")

    return results

# Optimized parallel processing
synthetic_labels = []
synthetic_features = []
clahe_images = []

# Parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    process_func = partial(
        process_patient_record_parallel,
        df=df,
        left_eye_keywords=left_eye_keywords,
        right_eye_keywords=right_eye_keywords,
        key_all=key_all,
        target_size=TARGET_SIZE
    )

    futures = [executor.submit(process_func, i) for i in range(len(df))]

    for future in concurrent.futures.as_completed(futures):
        row_results = future.result()
        for label, feature, clahe_img in row_results:
            if label is not None:
                synthetic_labels.append(label)
                synthetic_features.append(feature)
                clahe_images.append(clahe_img)

# %%
from sklearn.model_selection import train_test_split

clahe_images = np.stack(clahe_images, axis=0)
synthetic_labels = np.asarray(synthetic_labels)

training_features, tmp_validation_features, training_labels, tmp_validation_labels, training_filenames, tmp_validation_filenames = train_test_split(clahe_images, synthetic_labels, synthetic_features, test_size=0.102, random_state=1)
validation_features, validation_test_features, validation_labels, validation_test_labels, validation_filenames, validation_test_filenames = train_test_split(tmp_validation_features, tmp_validation_labels, tmp_validation_filenames, test_size=0.02, random_state=1)

print("n training:", len(training_filenames))
print("n validation:", len(validation_filenames))
print("n validation test:", len(validation_test_filenames))

# Delete temporary list file for minimalizing memory usage
del clahe_images
del synthetic_labels
del tmp_validation_features
del tmp_validation_labels
del tmp_validation_filenames

# %%
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
for idx in range(10):
    i, j = divmod(idx, 5)
    if COLOR_MODE == 'rgb':
        ax[i,j].imshow(training_features[idx].reshape(TARGET_SIZE[0], TARGET_SIZE[1], 3), cmap="hsv")
    else:
        ax[i,j].imshow(training_features[idx].reshape(TARGET_SIZE[0], TARGET_SIZE[1]), cmap="gray")

plt.tight_layout()

# %%
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
for idx in range(10):
    i, j = divmod(idx, 5)
    ax[i,j].imshow(validation_features[idx].reshape(TARGET_SIZE[0], TARGET_SIZE[1], 3), cmap="hsv")

plt.tight_layout()

# %%
# 1. Define Preprocessing/Augmentation Pipeline
augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=30/360, fill_mode='nearest'), # rotation_range=30
    tf.keras.layers.RandomFlip("horizontal"), # horizontal_flip=True
])

rescaling_layer = tf.keras.layers.Rescaling(1./255)

def prepare_dataset(features, labels, augment=False):
    # Create dataset from numpy arrays
    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    # Apply rescaling (features are 0-255 numpy arrays from CLAHE processing)
    ds = ds.map(lambda x, y: (rescaling_layer(tf.cast(x, tf.float32)), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Shuffle and apply augmentations only to training
        ds = ds.shuffle(buffer_size=min(len(features), 1000))
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    return ds.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

train_generator = prepare_dataset(training_features, training_labels, augment=True)
validation_generator = prepare_dataset(validation_features, validation_labels)

# %%
USE_MODEL = "using custom"
USE_PRETRAINED_MODEL = False
INPUT_SHAPE = TARGET_SIZE + SHAPE_ADD

N_EPOCH = 50
LEARNING_RATE = 1e-4
LOSS = "binary_crossentropy"
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)

MODEL_PATH = 'Trained_Models/ODIR-5K-Multi-Label/'
MODEL_SAVE_WEIGHTS = 'weight.h5'
MODEL_SAVE_NAME_H5 = 'ODIR5K.h5'

CHECKPOINT_PATH = MODEL_PATH + 'ODIR5K.keras'

AUC_VALUE = tf.keras.metrics.AUC(name='auc_value', curve='ROC', summation_method='interpolation', multi_label=True)
PRECISION_SCORE = tf.keras.metrics.Precision(thresholds=0.5, name='precision')
RECALL_SCORE = tf.keras.metrics.Recall(thresholds=0.5, name='recall')

STOP_ACCURACY = 0.90

class CallbackStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > STOP_ACCURACY):
            print("Reached", STOP_ACCURACY * 100, "accuracy so cancelling training!")
            self.model.stop_training = True

callback_stop = CallbackStop()
callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, verbose=1)

@tf.function
def accuracy_multilabel(y, y_hat):
    correct_prediction = tf.equal(tf.round(y_hat), tf.cast(y, tf.float32))
    # correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(y_hat)), tf.round(y))
    # mean
    correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # all
    # all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
    # correct_prediction = tf.reduce_mean(all_labels_true)
    return correct_prediction

@tf.function
def multilabel_cross_entropy(y, y_hat):
    # cross_entropy = -tf.reduce_sum(((y * tf.math.log(y_hat + 1e-9)) + ((1-y) * tf.math.log(1 - y_hat + 1e-9)) ), name='xentropy')
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat, name="sigmoid_cross_entropy_with_logits")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.cast(y,tf.float32))
    loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
    return loss

# %%
if os.path.isfile(MODEL_PATH + MODEL_SAVE_NAME_H5) and USE_PRETRAINED_MODEL:
    print("Using h5")
    model = tf.keras.models.load_model(MODEL_PATH + MODEL_SAVE_NAME_H5)
else:
    print("No using saved model")
    if USE_MODEL == "using custom":
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
        inputs = tf.keras.Input(shape=INPUT_SHAPE)

        # The first convolution block
        x = Conv2D(32, (3,3), activation='relu')(inputs)
        x = Conv2D(32, (3,3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = BatchNormalization()(x)

        # The second convolution block
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = BatchNormalization()(x)

        # The third convolution block
        x = Conv2D(128, (3,3), activation='relu')(x)
        x = Conv2D(128, (3,3), activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = BatchNormalization()(x)

        # The fourth convolution block
        x = Conv2D(256, (3,3), activation='relu')(x)
        x = Conv2D(256, (3,3), activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = BatchNormalization()(x)

        # Dense layers
        x = Flatten()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(8, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary(line_length=100)
model.compile(loss=LOSS,
              optimizer=OPTIMIZER,
              metrics=[accuracy_multilabel, AUC_VALUE, PRECISION_SCORE, RECALL_SCORE])

# %%
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=N_EPOCH,
                    verbose=1,
                    callbacks=[callback_stop])

# %%
model.save_weights(MODEL_PATH + MODEL_SAVE_WEIGHTS)
model.save(MODEL_PATH + MODEL_SAVE_NAME_H5)

# %%
metrics = [
    ('accuracy_multilabel', 'accuracy', 0),
    ('loss', 'loss', 1),
    ('auc', 'AUC value', 3),
    ('precision', 'Precision', 2),
    ('recall', 'Recall', 4)
]
epochs = range(1, len(history.history['loss']) + 1)
for key, label, loc in metrics:
    plt.plot(epochs, history.history[key], 'r', label=f'Training {label}')
    plt.plot(epochs, history.history[f'val_{key}'], 'y', label=f'Validation {label}')
    plt.title(f'Training and validation {label}')
    plt.legend(loc=loc)
    plt.figure()
plt.show()

# %%
TRAINING_SOURCE_PATH = 'ODIR-5K_Training_Images/'
TESTING_SOURCE_PATH = 'ODIR-5K_Tesing_Images/'

# output = tf.metrics.MultiLabelConfusionMatrix(num_classes=8)
print("\nVALIDATION TEST RESULTS")
print(f"{'File Name':<30} {'True Label':<15} {'Predicted':<15} {'Accuracy':<10}")

count_true = 0
count_half = 0
count_zero = 0

for i in range(len(validation_test_filenames)):
    source = TRAINING_SOURCE_PATH + validation_test_filenames[i]
    img = CLAHE(source, TARGET_SIZE, 20, (10,10))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    images = np.vstack([img_array])
    predict = model.predict(images)
    predict = predict.reshape(8)
    predict = tf.cast(predict >= 0.5, np.int32)
    y_true = tf.constant(validation_test_labels[i], dtype=tf.int32)
    y_pred = tf.constant(predict.numpy(), dtype=tf.int32)
    acc_ml = accuracy_multilabel(y_true, y_pred).numpy()

    # Count results
    count_true = count_true + 1 if acc_ml == 1.0 else count_true
    count_half = count_half + 1 if 0.75 <= acc_ml < 1 and 1 in predict.numpy().tolist() else count_half
    count_zero = count_zero + 1 if 1 not in predict.numpy().tolist() else count_zero

    # Format and display results
    filename = os.path.basename(source)
    true_label = str(validation_test_labels[i])
    predicted = str(predict.numpy())
    accuracy = f"{acc_ml:.3f}"

    print(f"{filename:<30} {true_label:<15} {predicted:<15} {accuracy:<10}")

print(f"\nResults Summary: True: {count_true} | Half True: {count_half} | Zero: {count_zero}")

test_list = os.listdir(TESTING_SOURCE_PATH)
test_list.sort()
for i in range(0, len(test_list), 5):
    source = TESTING_SOURCE_PATH + test_list[i]
    img = CLAHE(source, TARGET_SIZE, 20, (10,10))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    images = np.vstack([img_array]) / 255
    classes = model.predict(images)
    classes = tf.cast(classes > 0.5, float)
    print(source, classes)