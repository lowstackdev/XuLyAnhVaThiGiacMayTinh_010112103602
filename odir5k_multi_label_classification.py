# %% [markdown]
# # Set Dependencies

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import tensorflow as tf
import tensorflow.keras.optimizers
import tensorflow.keras.backend as K

print(tf.__version__)

# %%
os.chdir('ODIR-5K')

# %% [markdown]
# # Data Preprocessing

# %%
from pandas import read_excel

file_name = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
df = read_excel(file_name)
print(df.head())

# %%
left_eye_keywords = df['Left-Diagnostic Keywords'].copy()
right_eye_keywords = df['Right-Diagnostic Keywords'].copy()

left_eye_keywords = left_eye_keywords.str.split("，").apply(lambda x: list(set(x)))
right_eye_keywords = right_eye_keywords.str.split("，").apply(lambda x: list(set(x)))

print(left_eye_keywords[2])

# %% [markdown]
# ## set the different keyword diagnosis label

# %%
mlb = MultiLabelBinarizer()

combined_keywords = pd.concat([left_eye_keywords, right_eye_keywords])
mlb.fit(combined_keywords)

all_diagnosis = list(mlb.classes_)
print("Total different keys diagnosis:", len(all_diagnosis))

# %% [markdown]
# ## get keywords from single label

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

label_string = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
key_all = [key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease]

for i in range(8):
	print(label_string[i], len(key_all[i]))

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
    print(label_string[i], len(key_all[i]))

print("Intersect by other:")
for i in range(len(key_all)):
    print(label_string[i], len(key_all[i]))

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
print("Double lablel row", len(double_diagnosis_row))
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
# ## get keywords from multilabel

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

# %% [markdown]
# ## add non recognized label to a label with the likelihood of approaching

# %%
#manual listing key
keywords_to_process = [
    ('suspected cataract', 3),
    ('image offset', 4)
]
for keyword, disease_group_index in keywords_to_process:
    if keyword in unrecognized_keywords_list and keyword not in all_key_diagnosis:
        key_all[disease_group_index].append(keyword)
        unrecognized_keywords_list.remove(keyword)

print(key_all[4])

# %%
[print("Not in:", keyword) for keyword in unrecognized_keywords_list if keyword not in all_key_diagnosis]
string = 'central serous chorioretinopathy'
print(string in key_other_disease)

# %%
# Set path
training_source_path = 'ODIR-5K_Training_Images/'
testing_source_path = 'ODIR-5K_Tesing_Images/'

training_path = 'training/'
validation_path = 'validation/'
testing_path = 'testing/'

# %% [markdown]
# ## Image processing

# %%
# Define croping function with tensorflow resize
def crop_image(image_path):
	image_data = tf.keras.preprocessing.image.load_img(image_path)
	array = tf.keras.preprocessing.image.img_to_array(image_data)
	image = tf.image.resize(array, [200,200], method='bilinear', preserve_aspect_ratio=True, antialias=False)
	# image = image / 255.0
	return image

# Define method for image resize, croping and image Contrast Limited Adaptive Histogram Equalization (CLAHE)
# using Opencv 4

def image_resize(image_path, dim):
	img = cv2.imread(image_path)
	if img.shape[1] != img.shape[0]:
		x = img.shape[1] // 2
		y = img.shape[0] // 2
		x = x-y
		img = img[0:0 + img.shape[0], x:x + img.shape[0]]
	# resize image
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
# Before CLAHE processing
source = 'ODIR-5K_Training_Images/441_left.jpg'
test = crop_image(source)
test = np.array(test)
img = tf.keras.preprocessing.image.array_to_img(test)
plt.imshow(img)
test = np.expand_dims(test, axis=0)
print(test.shape)

# %%
# Showing CLAHE image Preprocessing
source = 'ODIR-5K_Training_Images/441_left.jpg'
test = CLAHE(source, (200,200), 20, (10,10))
test = np.array(test)
img = tf.keras.preprocessing.image.array_to_img(test)
plt.imshow(img)
test = test.reshape(1, 200, 200, 3)
print(test.shape)

# %%
# Set target size image

target_size = (230, 230)
# color_mode = 'grayscale'
color_mode = 'rgb'
if color_mode == 'grayscale':
	shape_add = (1,)
if color_mode == 'rgb':
	shape_add = (3,)

# %% [markdown]
# ## Synthetizing the label for single image

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
		else :
			tmp_label.append(0)
	return tmp_label

# %%
synthetic_labels = []
synthetic_features = []
clahe_images = []
for i in range(len(df)):
    try:
        left_img_path = os.path.join('ODIR-5K_Training_Images', df['Left-Fundus'][i])
        right_img_path = os.path.join('ODIR-5K_Training_Images', df['Right-Fundus'][i])

        left_fundus_img = cv2.imread(left_img_path)
        right_fundus_img = cv2.imread(right_img_path)

        if left_fundus_img is None or right_fundus_img is None:
            print(f"Warning: Could not read images at row {i}")
            continue
        try:
            left_indices = [get_index_label(key, key_all) for key in left_eye_keywords[i]]
            left_indices = list(set(left_indices))
            synthetic_labels.append(get_multi_label_from_keys(left_indices))
            synthetic_features.append(df['Left-Fundus'][i])
            clahe_images.append(CLAHE(left_img_path, target_size, 20, (10,10)))

            right_indices = [get_index_label(key, key_all) for key in right_eye_keywords[i]]
            right_indices = list(set(right_indices))
            synthetic_labels.append(get_multi_label_from_keys(right_indices))
            synthetic_features.append(df['Right-Fundus'][i])
            clahe_images.append(CLAHE(right_img_path, target_size, 20, (10,10)))

        except Exception as e:
            print(f"Error processing keywords at row {i}: {str(e)}")
            continue

    except Exception as e:
        print(f"Error reading images at row {i}: {str(e)}")
        continue

# %% [markdown]
# ## split feature, label, and file name for training, validation and test

# %%
import numpy as np
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

# %% [markdown]
# ## show some image for training

# %%
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
for idx in range(10):
    i, j = divmod(idx, 5)
    if color_mode == 'rgb':
        ax[i,j].imshow(training_features[idx].reshape(target_size[0], target_size[1], 3), cmap="hsv")
    else:
        ax[i,j].imshow(training_features[idx].reshape(target_size[0], target_size[1]), cmap="gray")

plt.tight_layout()

# %% [markdown]
# ## show some image for validation

# %%
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
for idx in range(10):
    i, j = divmod(idx, 5)
    ax[i,j].imshow(validation_features[idx].reshape(target_size[0], target_size[1], 3), cmap="hsv")

plt.tight_layout()

# %% [markdown]
# ## set image data generator for training

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

# %% [markdown]
# # Train

# %% [markdown]
# ## Set callback method

# %%
checkpoint_path = "Trained_Models/ODIR5K-Multi-Label/ODIR5K.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
												 # save_weights_only=True,
												 verbose=1)

stop_val_auc = 0.8200
stop_accuracy = 0.90
stop_val_accuracy = 0.90

# Define a Callback class that stops training once accuracy reaches the certain accuracy
class CallbackStop(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy_multilabel', 0.0) > stop_accuracy or logs.get('val_accuracy_multilabel', 0.0) > stop_val_accuracy):
			print(f"Reached accuracy threshold ({stop_accuracy}) or validation accuracy threshold ({stop_val_accuracy}) so cancelling training!")
			self.model.stop_training = True

callback_stop = CallbackStop()

# %% [markdown]
# ## Set metric for training

# %%
auc_value = tf.keras.metrics.AUC(name='auc_value',
                                  # num_thresholds=200,
                                  curve='ROC',
                                  summation_method='interpolation',
                                  # thresholds=0.5,
                                  multi_label=True)

precision_score = tf.keras.metrics.Precision(thresholds=0.5, name='precision')
recall_score = tf.keras.metrics.Recall(thresholds=0.5, name='recall')

# %%

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
def accuracy_multilabel2(y, y_hat):
	correct_prediction = tf.equal(tf.round(y_hat), tf.round(tf.cast(y, tf.float32)))
	# correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(y_hat)), tf.round(y))
	# mean
	# correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# all
	correct_prediction = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
	correct_prediction = tf.reduce_mean(correct_prediction)
	return correct_prediction

@tf.function
def exact_match_fn(y_true, y_logits):
	threshold=0.5
	#pred = tf.equal(tf.round(y_logits), tf.round(y_true))
	predictions = tf.cast(tf.greater_equal(y_logits, threshold), dtype=tf.float32)
	pred_match = tf.equal(predictions, tf.round(y_true))
	exact_match = tf.reduce_min(tf.cast(pred_match, dtype=tf.float32), axis=1)
	return exact_match

@tf.function
def exact_match_prop_fn(*args):
	return tf.reduce_mean(exact_match_fn(*args))

class MetricsAtTopK:
	def __init__(self, k):
		self.k = k

	def _get_prediction_tensor(self, y_pred):
		"""Takes y_pred and creates a tensor of same shape with 1 in indices where, the values are in top_k
		"""
		topk_values, topk_indices = tf.nn.top_k(y_pred, k=self.k, sorted=False, name="topk")
		# the topk_indices are along last axis (1). Add indices for axis=0
		ii, _ = tf.meshgrid(tf.range(tf.shape(y_pred)[0]), tf.range(self.k), indexing='ij')
		index_tensor = tf.reshape(tf.stack([ii, topk_indices], axis=-1), shape=(-1, 2))
		prediction_tensor = y_pred
		# prediction_tensor =  tf.sparse.to_dense(sparse_indices=index_tensor, output_shape=tf.shape(y_pred), default_value=0, sparse_values=1.0, validate_indices=False)
		prediction_tensor = tf.cast(prediction_tensor, K.floatx())
		return prediction_tensor

	def true_positives_at_k(self, y_true, y_pred):
		prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
		y_true = tf.cast(y_true, tf.float32)
		true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
		return true_positive

	def false_positives_at_k(self, y_true, y_pred):
		prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
		y_true = tf.cast(y_true, tf.float32)
		true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
		c2 = K.sum(prediction_tensor)  # TP + FP
		false_positive = c2 - true_positive
		return false_positive

	def false_negatives_at_k(self, y_true, y_pred):
		prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
		y_true = tf.cast(y_true, tf.float32)
		true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
		c3 = K.sum(y_true)  # TP + FN
		false_negative = c3 - true_positive
		return false_negative

	def precision_at_k(self, y_true, y_pred):
		prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
		y_true = tf.cast(y_true, tf.float32)
		true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
		c2 = K.sum(prediction_tensor)  # TP + FP
		return true_positive / (c2 + K.epsilon())

	def recall_at_k(self, y_true, y_pred):
		prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
		y_true = tf.cast(y_true, tf.float32)
		true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
		c3 = K.sum(y_true)  # TP + FN
		return true_positive / (c3 + K.epsilon())

metrics_at_top_k = MetricsAtTopK(k=5)

@tf.function
def hamming_loss(y_true, y_pred, mode='multiclass'):
	if mode not in ['multiclass', 'multilabel']:
		raise TypeError('mode must be: [multiclass, multilabel])')

	if mode == 'multiclass':
		nonzero = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=-1), tf.float32)
		print(nonzero)
		return 1.0 - nonzero

	else:
		nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
		return nonzero / y_true.get_shape()[-1]

class HammingLoss(tf.keras.metrics.MeanMetricWrapper):
	def __init__(self, name='hamming_loss', dtype=None, mode='multiclass'):
		super(HammingLoss, self).__init__(hamming_loss, name, dtype=dtype, mode=mode)

hl_metric = HammingLoss()

# %% [markdown]
# ## set loss function for training

# %%

@tf.function
def multilabel_cross_entropy(y, y_hat):
	# cross_entropy = -tf.reduce_sum(((y * tf.math.log(y_hat + 1e-9)) + ((1-y) * tf.math.log(1 - y_hat + 1e-9)) ), name='xentropy')
	# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat, name="sigmoid_cross_entropy_with_logits")
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.cast(y,tf.float32))
	loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
	return loss

@tf.function
def npairs_multilabel_loss(y_true, y_pred):
	y_pred = tf.matmul(y_true, y_pred, transpose_a=False, transpose_b=True)
	loss = tf.losses.npairs_multilabel_loss(y, y_pred)
	return loss

@tf.function
def hamming_loss_func(y_true, y_pred):
	diff = tf.cast(y_true - y_pred, dtype=tf.float32)

	#Counting non-zeros in a differentiable way
	epsilon = K.epsilon()
	nonzero = tf.reduce_sum(tf.math.abs(diff / (tf.math.abs(diff) + epsilon)))

	return tf.reduce_mean(nonzero / K.int_shape(y_pred)[-1])

# %% [markdown]
# ## Define Model

# %%
# use_model = "use transfer learning using vgg19"
# use_model = "use transfer learning using mobilenetv2"
use_model = "using custom"

# %%
if ("use transfer learning" in use_model):
	if("using vgg19" in use_model):
		base_model= tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=target_size + shape_add,)
		model_path = 'Trained_Models/ODIR-5K-VGG19-Multi-Label/'

	if("using mobilenetv2" in use_model):
		base_model= tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=target_size + shape_add,)
		model_path = 'Trained_Models/ODIR-5K-MobileNetV2-Multi-Label/'

	for layer in base_model.layers:
		layer.trainable = False

	base_model.summary(line_length=100)
	# last_layer = base_model.get_layer('block4_pool')
	# conn = last_layer.output

	conn = base_model.output

	conn = tf.keras.layers.Flatten()(conn)
	conn = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.2)(conn)
	conn = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	# conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)
	# conn = tf.keras.layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	# conn = tf.keras.layers.BatchNormalization()(conn)
	conn = tf.keras.layers.Dropout(0.2)(conn)
	# conn = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	# conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)
	conn = tf.keras.layers.Dense(8, activation='sigmoid')(conn)
if (use_model == "using custom"):
	model_path = 'Trained_Models/ODIR-5K-VGG16_like-Multi-Label/'

	inputs = tf.keras.Input(shape=target_size + shape_add)
	# The first convolution
	conn = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
	conn = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.MaxPooling2D(2, 2)(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)

	# The second convolution
	conn = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.MaxPooling2D(2,2)(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)

	# The third convolution
	conn = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.MaxPooling2D(2,2)(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)

	# The fourth convolution
	conn = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(conn)
	conn = tf.keras.layers.MaxPooling2D(2,2)(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)

	conn = tf.keras.layers.Flatten()(conn)
	conn = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)
	conn = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)
	# conn = tf.keras.layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(conn)
	# conn = tf.keras.layers.BatchNormalization()(conn)
	# conn = tf.keras.layers.Dropout(0.2)(conn)
	conn = tf.keras.layers.Dense(64, activation='relu',
	# kernel_regularizer=tf.keras.regularizers.l2(0.001)
	)(conn)
	# conn = tf.keras.layers.Dropout(0.4)(conn)
	conn = tf.keras.layers.Dense(8, activation='sigmoid')(conn)

# %% [markdown]
# training dataset

# %%
use_training_model = False

checkpoint_path = model_path + 'ODIR5K.keras'
checkpoint_dir = os.path.dirname(checkpoint_path)
model_save_weights = 'weight'
model_save_name_h5 = 'ODIR5K.h5'
model_save_name_tf = 'ODIR5K_TF'
model_save_name_js = 'ODIR5K_TFJS'

n_epoch = 25
# target_size = (200,200)
# shape_add = (3,)
input_shape = target_size + shape_add
learning_rate = 1e-4
# loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss = "binary_crossentropy"
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# tf.keras.optimizers.SGD(learning_rate=learning_rate)

if (os.path.isfile(model_path + model_save_name_h5) or os.path.exists(model_path + model_save_name_tf)) and use_training_model:
	# if os.path.exists(model_path + model_save_name_tf):
	# 	print("Using tf")
	# 	model = tf.keras.models.load_model(model_path + model_save_name_tf)
	if os.path.isfile(model_path + model_save_name_h5):
		print("Using h5")
		model = tf.keras.models.load_model(model_path + model_save_name_h5)
	output = model.output
else:
	print("No using saved model")
	if (use_model == "using custom"):
		model = tf.keras.Model(inputs=inputs, outputs=conn)
	else:
		model = tf.keras.Model(base_model.input, conn)
	# model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary(line_length=100)
model.compile(loss='binary_crossentropy',
              # macro_soft_f1, # not suitable loss for multilabel
              # npairs_multilabel_loss, #cannot use in this model
              # hamming_loss_func
			  optimizer=optimizer,
			  metrics=[accuracy_multilabel, auc_value, precision_score, recall_score])

# %%
history = model.fit(train_generator,
					# train_generator_noaugment,
					validation_data=validation_generator,
					epochs=50,
					# steps_per_epoch=100,
					# batch_size=train_generator.batch_size,
					# steps_per_epoch = train_generator.samples // train_generator.batch_size,
					# validation_steps = validation_generator.samples // validaition_generator.batch_size,
					verbose=1,
					callbacks=[callback_stop,
							# cp_callback
							])

# %% [markdown]
# # Evaluate

# %% [markdown]
# ## Validation test trained model

# %%
training_path = 'ODIR-5K_Training_Images/'
training_list = os.listdir('ODIR-5K_Training_Images/')
output = tf.metrics.MultiLabelConfusionMatrix(num_classes=8)
print("file name", "\t\t\t\t\t", "true label", "\t\t", "prediction label","\t", "accuracy score")
count_true = 0
count_half = 0
count_zero = 0
for i in range(0, len(validation_test_filenames)):
	source = training_path + validation_test_filenames[i]
	# img = tf.keras.preprocessing.image.load_img(source, target_size=target_size,)
	img = CLAHE(source, target_size, 20, (10,10))
	# cv_imshow(img)
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	img_array = img_array/255.0
	images = np.vstack([img_array])
	predict = model.predict(images)
	# print(predict.shape)
	predict = predict.reshape(8)
	# y_true = validation_test_labels[i].reshape(1,8)
	predict = tf.cast(predict >= 0.5, np.int32)
	y_true = tf.constant(validation_test_labels[i], dtype=tf.int32)
	y_pred = tf.constant(predict.numpy(), dtype=tf.int32)
	acc_ml = accuracy_multilabel(y_true, y_pred).numpy()
	count_true = count_true + 1 if acc_ml == 1.0 else count_true
	count_half = count_half + 1 if 0.75 <= acc_ml < 1 and 1 in predict.numpy().tolist() else count_half
	count_zero = count_zero + 1 if 1 not in predict.numpy().tolist() else count_zero
	# output.update_state(y_true, y_pred)
	print("---------------------------------------------------------------------------------------------------")
	print(source, "\t", validation_test_labels[i], "\t", predict.numpy(), "\t", acc_ml)
	# print(output.result().numpy())

print('\n',"true:", count_true, "| half true:", count_half, "| zero:", count_zero)

# %% [markdown]
# ## save the trained model

# %%
model.save_weights(model_path)
model.save_weights(model_path + model_save_weights)
model.save(model_path)
model.save(model_path + model_save_name_h5)
model.save(model_path + model_save_name_tf,save_format='tf')

# %%
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, model_path + model_save_name_js)

# %%
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()
open(model_path + "ODIR5K.tflite", "wb").write(tflite_model)

# %% [markdown]
# ## plot the training and validation step

# %%
precision = history.history['precision']
val_precision = history.history['val_precision']

recall = history.history['recall']
val_recall = history.history['val_recall']

acc = history.history['accuracy_multilabel']
val_acc = history.history['val_accuracy_multilabel']

loss = history.history['loss']
val_loss = history.history['val_loss']

auc = history.history['auc_value']
val_auc = history.history['val_auc_value']

epochs_training = range(1, len(acc) + 1)

plt.plot(epochs_training, acc, 'r', label='Training accuracy')
plt.plot(epochs_training, val_acc, 'y', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.savefig(model_path + 'accuracy.png')

plt.plot(epochs_training, loss, 'r', label='Training loss')
plt.plot(epochs_training, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=1)
plt.figure()
plt.savefig(model_path + 'loss.png')

# plt.plot(epochs_training, kappa, 'r', label='Training kappa score')
# plt.plot(epochs_training, val_kappa, 'y', label='Validation kappa score')
# plt.title('Training and validation kappa score')
# plt.legend(loc=2)
# plt.figure()

plt.plot(epochs_training, auc, 'r', label='Training AUC value')
plt.plot(epochs_training, val_auc, 'y', label='Validation AUC value')
plt.title('Training and validation AUC value')
plt.legend(loc=3)
plt.figure()
plt.savefig(model_path + 'AUC.png')

plt.plot(epochs_training, precision, 'r', label='Training Precision')
plt.plot(epochs_training, val_precision, 'y', label='Validation Precision')
plt.title('Training and validation Precision')
plt.legend(loc=2)
plt.figure()

plt.plot(epochs_training, recall, 'r', label='Training Recall')
plt.plot(epochs_training, val_recall, 'y', label='Validation Recall')
plt.title('Training and validation Recall')
plt.legend(loc=4)
plt.savefig(model_path + 'precision_recall.png')

plt.show()

# %% [markdown]
# ## testing model

# %%
# for ifile in test_list:
test_path = 'ODIR-5K_Tesing_Images/'
test_list = os.listdir('ODIR-5K_Tesing_Images/')
test_list.sort()
# source = 'ODIR-5K_Training_Images/12_left.jpg'
for i in range(0, len(test_list), 5):
	source = test_path+test_list[i]
	# img = tf.keras.preprocessing.image.load_img(source, target_size=target_size,)
	img = CLAHE(source, target_size, 20, (10,10))
	# cv_imshow(img)
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	images = np.vstack([img_array])/255
	classes = model.predict(images)
	# print(classes, '\n')
	classes = tf.cast(classes > 0.5, float)
	print(source, classes)