# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
import cv2
import tensorflow as tf

print(tf.__version__)

# %%
PROJECT_ROOT = Path(__file__).parent.resolve()
# PROJECT_ROOT = '/content/drive/MyDrive/Colab Notebooks'
DATASET_DIR = PROJECT_ROOT / "ODIR-5K"
os.chdir(DATASET_DIR)

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

    unique_keywords = set().union(*[set(left_eye_keywords[row]) | set(right_eye_keywords[row])
                                    for row in single_rows])

    return list(unique_keywords)

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

# %%
# Define method for image resize, cropping and image Contrast Limited Adaptive Histogram Equalization (CLAHE)
# using Opencv 4

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
TARGET_SIZE = (230, 230)
COLOR_MODE = 'rgb'
COLOR_SHAPE_MAP = {
    'grayscale': (1,),
    'rgb': (3,),
    'rgba': (4,)
}
SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

# %%
# Function for generate label to single image

# Return index in key of all diagnosis list
def get_index_label(key, key_all):
    return next((i for i, keywords in enumerate(key_all) if key in keywords), -1)

# Return multilabel by index
def get_multi_label_from_keys(idx_label):
    return [1 if i in idx_label else 0 for i in range(8)]

import concurrent.futures
from functools import partial

def process_fundus_image_with_clahe(img_path, keywords, key_all, target_size):
    """Process a single fundus image with CLAHE enhancement and generate diagnostic labels"""
    try:
        # Read image and check if valid
        if (fundus_img := cv2.imread(img_path)) is None:
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
        # Process both eyes using a loop
        for eye_side, fundus_col, keywords_col in [('Left', 'Left-Fundus', left_eye_keywords), ('Right', 'Right-Fundus', right_eye_keywords)]:
            img_path = os.path.join(TRAINING_SOURCE_PATH, df[fundus_col][row_idx])
            label, feature, clahe = process_fundus_image_with_clahe(img_path, keywords_col[row_idx], key_all, target_size)
            if label is not None:
                results.append((label, feature, clahe))

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
        for label, feature, clahe_img in future.result():
            if label is not None:
                synthetic_labels.append(label)
                synthetic_features.append(feature)
                clahe_images.append(clahe_img)

# %%
from sklearn.model_selection import train_test_split, GroupShuffleSplit

clahe_images = np.stack(clahe_images, axis=0)
synthetic_labels = np.asarray(synthetic_labels)

# Grouping by patient ID to prevent data leakage (same patient's eyes in different sets)
groups = [f.split('_')[0] for f in synthetic_features]

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
train_idx, val_idx = next(gss.split(clahe_images, synthetic_labels, groups=groups))

training_features = clahe_images[train_idx]
training_labels = synthetic_labels[train_idx]
training_filenames = [synthetic_features[i] for i in train_idx]

validation_features = clahe_images[val_idx]
validation_labels = synthetic_labels[val_idx]
validation_filenames = [synthetic_features[i] for i in val_idx]

print("n training:", len(training_filenames))
print("n validation:", len(validation_filenames))

# Approximate class weights by looking at the presence of each class (multi-label)
train_labels_idx = np.argmax(training_labels, axis=1)
class_weights_vals = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(8),
    y=train_labels_idx
)
class_weights = dict(enumerate(class_weights_vals))

del clahe_images
del synthetic_labels

# %%
def display_image_samples(features, title, color_mode, target_size):
    """Display a 2x5 grid of image samples with proper coloring based on color mode"""
    f, ax = plt.subplots(2, 5)
    f.set_size_inches(10, 10)
    f.suptitle(title, fontsize=16)

    for idx in range(10):
        i, j = divmod(idx, 5)
        if color_mode == 'rgb':
            ax[i,j].imshow(features[idx].reshape(target_size[0], target_size[1], 3), cmap="hsv")
        else:
            ax[i,j].imshow(features[idx].reshape(target_size[0], target_size[1]), cmap="gray")

    plt.tight_layout()
    plt.show()

display_image_samples(training_features, "Training Image Samples", COLOR_MODE, TARGET_SIZE)
display_image_samples(validation_features, "Validation Image Samples", COLOR_MODE, TARGET_SIZE)

# %%
# 1. Define Preprocessing/Augmentation Pipeline
augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.0833, fill_mode='nearest'),  # ±30 degrees
    tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode='nearest'),
    tf.keras.layers.RandomBrightness(factor=0.1),  # Adjust brightness
    tf.keras.layers.RandomContrast(factor=0.1),    # Adjust contrast
    tf.keras.layers.GaussianNoise(0.01),
])

rescaling_layer = tf.keras.layers.Rescaling(1./255)

# 2. Prepare Dataset for training and validation
def prepare_dataset(features, labels, batch_size=32, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    # Apply rescaling (features are 0-255 numpy arrays from CLAHE processing)
    ds = ds.map(lambda x, y: (rescaling_layer(tf.cast(x, tf.float32)), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()

    if augment:
        # Shuffle and apply augmentations only to training
        ds = ds.shuffle(buffer_size=min(len(features), 1000))
        # ds = ds.repeat() # Removed to allow automatic step calculation
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

BATCH_SIZE = 32
train_generator = prepare_dataset(training_features, training_labels, batch_size=BATCH_SIZE, augment=True)
validation_generator = prepare_dataset(validation_features, validation_labels, batch_size=BATCH_SIZE)

# %%
@tf.function
def accuracy_multilabel(y, y_hat):
    y_float = tf.cast(y, tf.float32)
    y_hat = tf.round(y_hat)
    correct_prediction = tf.equal(y_hat, y_float)
    # mean
    correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_prediction

@tf.function
def multilabel_cross_entropy(y, y_hat):
    # cross_entropy = -tf.reduce_sum(((y * tf.math.log(y_hat + 1e-9)) + ((1-y) * tf.math.log(1 - y_hat + 1e-9)) ), name='xentropy')
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat, name="sigmoid_cross_entropy_with_logits")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.cast(y,tf.float32))
    loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
    return loss

USE_MODEL = "using custom"
USE_PRETRAINED_MODEL = False
INPUT_SHAPE = TARGET_SIZE + SHAPE_ADD

N_EPOCH = 1
LEARNING_RATE = 1e-4
LOSS = "binary_crossentropy"
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)

ACCURACY_SCORE = 'accuracy_multilabel'
AUC_VALUE = tf.keras.metrics.AUC(name='auc_value', curve='ROC', summation_method='interpolation', multi_label=True)
PRECISION_SCORE = tf.keras.metrics.Precision(thresholds=0.5, name='precision')
RECALL_SCORE = tf.keras.metrics.Recall(thresholds=0.5, name='recall')

MODEL_DIR = PROJECT_ROOT / "Trained_Models" / "ODIR-5K-Multi-Label"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_WEIGHTS = str(MODEL_DIR / 'ODIR5K_weights.weights.h5')
MODEL_SAVE_FINAL = str(MODEL_DIR / 'ODIR5K_final.keras')
CHECKPOINT_PATH = str(MODEL_DIR / 'ODIR5K.keras')

callbacks = [
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
  tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor='val_auc_value', save_best_only=True, mode='max', verbose=1),
  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

# %%
if os.path.isfile(str(MODEL_SAVE_FINAL)) and USE_PRETRAINED_MODEL:
    print("Using saved model")
    model = tf.keras.models.load_model(str(MODEL_SAVE_FINAL))
else:
    print("No using saved model")
    if USE_MODEL == "using custom":
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
        inputs = tf.keras.Input(shape=INPUT_SHAPE)

        # Conv larger kernel
        x = Conv2D(32, (7,7), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2,2)(x)

        # Residual blocks
        for filters in [64, 128, 256]:
            # Shortcut
            shortcut = Conv2D(filters, (1,1), padding='same')(x) if x.shape[-1] != filters else x

            # Conv block
            x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)

            # Add shortcut
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            x = MaxPooling2D(2,2)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        # Dense
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)

        outputs = Dense(8, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary(line_length=100)
model.compile(loss=LOSS,
              optimizer=OPTIMIZER,
              metrics=[ACCURACY_SCORE, AUC_VALUE, PRECISION_SCORE, RECALL_SCORE])

# %%
history = model.fit(train_generator=train_generator,
                    validation_data=validation_generator,
                    epochs=N_EPOCH,
                    class_weight=class_weights,
                    verbose=1,
                    callbacks=callbacks)

# %%
model.save_weights(MODEL_SAVE_WEIGHTS)
model.save(MODEL_SAVE_FINAL)

# %%
metrics = [
    ('accuracy_multilabel', 'accuracy', 0),
    ('loss', 'loss', 1),
    ('auc_value', 'AUC value', 3),
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
# %% [markdown]
# ## Combined Testing on Test Dataset

# %%
print(f"{'File Name':<30} {'Predicted Classes':<30} {'Predicted Labels':<20}")

test_list = sorted(f for f in os.listdir(TESTING_SOURCE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

print(f"Total testing images found: {len(test_list)}")

count_normal = 0
count_single_disease = 0
count_multiple_diseases = 0
count_no_disease = 0

for i in range(len(test_list)):
    source = os.path.join(TESTING_SOURCE_PATH, test_list[i])
    img = CLAHE(source, TARGET_SIZE, 20, (10,10))
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
    active_labels = [s for s, p in zip(LABEL_STRINGS, predicted_labels) if p] or ["None"]

    # Format and display results
    filename = os.path.basename(source)
    predicted_str = str(predicted_labels)
    labels_str = ', '.join(active_labels)

    print(f"{filename:<30} {labels_str:<30} {predicted_str:<20}")

print(f"\nTest Results Summary:")
print(f"Total Tested: {len(test_list)}")
print(f"Normal: {count_normal} | Single Disease: {count_single_disease} | Multiple Diseases: {count_multiple_diseases} | No Disease Detected: {count_no_disease}")
