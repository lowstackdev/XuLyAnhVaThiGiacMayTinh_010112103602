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

# %%
left_eye_keywords = left_eye_keywords.str.split("，").apply(lambda x: list(set(x)))
right_eye_keywords = right_eye_keywords.str.split("，").apply(lambda x: list(set(x)))

# %%
print(left_eye_keywords[2])

# %% [markdown]
# ## set the different keyword diagnosis label

# %%
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

all_diagnosis=list(set(all_diagnosis_left + all_diagnosis_right))
print("Total different keys diagnosis:", len(all_diagnosis))

# %% [markdown]
# ## get keywords from single label

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
	key_all[i] = list(set(key_all[i])-set(key_all[0]))

for i in range(8):
	print(label_string[i], len(key_all[i]))

print("Intersect by other:")
for i in range(len(key_all)):
	for j in range(i,len(key_all)):
		if i == j:
			continue
		else :
			key_all[i] = list(set(key_all[i])-set(key_all[j]))

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
len(all_key_diagnosis)

# %%
double_diagnosis_row = list(set(double_diagnosis_row))
print("Double lablel row",len(double_diagnosis_row))
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

# %% [markdown]
# ## get keywords from multilabel

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
				col_index.append(i-7)
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
			mall_key_diagnosis = mall_key_diagnosis + list(set(i))
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

# %% [markdown]
# ## add non recognized label to a label with the likelihood of approaching

# %%
#manual listing key

string = 'suspected cataract'
string2 = 'image offset'
if string in not_recognized_list and string not in all_key_diagnosis:
	key_all[3].append(string)
	not_recognized_list.remove(string)

if string2 in not_recognized_list and string not in all_key_diagnosis:
	print(True)
	key_all[4].append(string2)
	not_recognized_list.remove(string2)

# %%
print(key_all[4])

# %%
all_key_diagnosis = get_all_recognized_key(key_all)

key_normal, key_diabetes, key_glaucoma, key_cataract, key_amd, key_hypertension, key_myopia, key_other_disease = key_all[0], key_all[1], key_all[2], key_all[3], key_all[4], key_all[5], key_all[6], key_all[7]

for i in range(len(key_all)):
	print(label_string[i], len(key_all[i]))

print("All regnized key:",len(all_key_diagnosis))
print("Not recognized key:", list(set(all_diagnosis)-set(all_key_diagnosis)))

# %%
string = 'central serous chorioretinopathy'

for i in not_recognized_list:
	if i not in all_key_diagnosis:
		print("Not in:", i)

print(string in key_other_disease)

# %%
# Set path
train_dir = 'training'
validation_dir = 'validation'
test_dir = 'testing'

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

# %%
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
def get_index_label(key, m_key_all):
	# print(key)
	for i_list in m_key_all:
		if key in i_list:
			# print(m_key_all.index(i_list))
			return m_key_all.index(i_list)

# Return multilabel by index
def get_multi_label_from_keys(index_label):
	temp_label = []
	for i in range(8):
		if i in index_label:
			temp_label.append(1)
		else :
			temp_label.append(0)
	return temp_label
	# print(right_label)

# %%
syntetic_labels = []
syntetic_features = []
image_array = []
processed_image_array = []
clahe_image = []
list_clahe = []
path = 'ODIR-5K_Training_Images/'
for i in range(len(df)):
	left_fundus_img = cv2.imread(path + df['Left-Fundus'][i])
	right_fundus_img = cv2.imread(path + df['Right-Fundus'][i])
	if left_fundus_img is None or right_fundus_img is None:
		continue
	one_hot_index_left = []
	one_hot_index_right = []
	for left_key in left_eye_keywords[i]:
		one_hot_index_left.append(get_index_label(left_key, key_all))
	for right_key in right_eye_keywords[i]:
		one_hot_index_right.append(get_index_label(right_key, key_all))
	one_hot_index_left = list(set(one_hot_index_left))
	one_hot_index_right = list(set(one_hot_index_right))
	syntetic_labels.append(get_multi_label_from_keys(one_hot_index_left))
	syntetic_labels.append(get_multi_label_from_keys(one_hot_index_right))
	syntetic_features.append(df['Left-Fundus'][i])
	syntetic_features.append(df['Right-Fundus'][i])
	clahe_image.append(CLAHE(path + df['Left-Fundus'][i], target_size, 20, (10,10)))
	clahe_image.append(CLAHE(path + df['Right-Fundus'][i], target_size, 20, (10,10)))

# %% [markdown]
# ## split feature, label, and file name for training, validation and test

# %%
import numpy as np
from sklearn.model_selection import train_test_split

clahe_image = np.stack(clahe_image, axis=0)
syntetic_labels = np.asarray(syntetic_labels)

training_features, temp_validation_features, training_labels, temp_validation_labels, training_filenames, temp_validation_filenames = train_test_split(clahe_image, syntetic_labels, syntetic_features, test_size=0.102, random_state=1)

validation_features, validation_test_features, validation_labels, validation_test_labels, validation_filenames, validation_test_filenames = train_test_split(temp_validation_features, temp_validation_labels, temp_validation_filenames, test_size=0.02, random_state=1)

print("n training :", len(training_filenames))
print("n validation :", len(validation_filenames))
print("n validation test :", len(validation_test_filenames))

# Delete temporary list file for minimalizing memory usage
del clahe_image
del syntetic_labels
del temp_validation_features
del temp_validation_labels
del temp_validation_filenames

# %% [markdown]
# ## show some image for training

# %%
f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
	for j in range(5):
		if color_mode == 'rgb':
			ax[i,j].imshow(training_features[k].reshape(target_size[0], target_size[1], 3) , cmap = "hsv")
		else :
			ax[i,j].imshow(training_features[k].reshape(target_size[0], target_size[1]) , cmap = "gray")
		k += 1
	plt.tight_layout()

# %% [markdown]
# ## show some image for validation

# %%
f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
	for j in range(5):
		ax[i,j].imshow(validation_features[k].reshape(target_size[0], target_size[1], 3) , cmap = "hsv")
		k += 1
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

# Define a Callback class that stops training once accuracy reaches the certain accuracy
class CallbackStop(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy_multilabel', 0.0) > stop_accuracy or logs.get('val_accuracy_multilabel', 0.0) > stop_val_accuracy):
			print("Reached stoping value so cancelling training!")
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
	# cross_entropy = -tf.reduce_sum(((y*tf.math.log(y_hat + 1e-9)) + ((1-y) * tf.math.log(1 - y_hat + 1e-9)) ), name='xentropy')
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
	nonzero = tf.reduce_sum( tf.math.abs(diff / (tf.math.abs(diff) + epsilon) ))

	return tf.reduce_mean(nonzero / K.int_shape(y_pred)[-1])

# %% [markdown]
# ##Define Model

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
# ##plot the training and validation step

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