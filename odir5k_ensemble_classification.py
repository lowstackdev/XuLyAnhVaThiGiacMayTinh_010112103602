# %%
import os
from pathlib import Path
import shutil
from abc import ABCMeta, abstractmethod
from typing import Optional

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
import cv2

from odir5k_multi_class_classification import config as odir5kmcc
from odir5k_multi_label_classification import config as odir5kmlc

# %%
class TaskWeightScheduler(tf.keras.callbacks.Callback):
    """
    Implements Dynamic Weight Averaging (DWA) to balance task losses.
    Based on: https://arxiv.org/abs/1803.10704
    """
    def __init__(self, task_names, temperature=2.0):
        super(TaskWeightScheduler, self).__init__()
        self.task_names = task_names
        self.temperature = temperature
        self.loss_history = {name: [] for name in task_names}
        self.task_weights = {name: 1.0 for name in task_names}

    def on_epoch_end(self, epoch, logs=None):
        # 1. Update loss history
        for name in self.task_names:
            loss_val = logs.get(f"{name}_loss")
            if loss_val is not None:
                self.loss_history[name].append(loss_val)

        # 2. DWA after the first epoch (at least 2 epochs of loss)
        if epoch >= 1 and all(len(h) >= 2 for h in self.loss_history.values()):
            rs = []
            for name in self.task_names:
                r = self.loss_history[name][-1] / self.loss_history[name][-2]
                rs.append(r)

            rs = np.array(rs)
            exp_rs = np.exp(rs / self.temperature)
            new_weights = (len(self.task_names) * exp_rs) / np.sum(exp_rs)

            for i, name in enumerate(self.task_names):
                self.task_weights[name] = float(new_weights[i])

            print(f"\n--- Epoch {epoch+1}: DWA updated task weights ---")
            for name, weight in self.task_weights.items():
                print(f"  - {name}: {weight:.4f}")

            # 3. Update model loss weights by re-compiling
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                loss_weights=self.task_weights,
                metrics=config.METRICS
            )

# %%
class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.resolve()
    try:
        import google.colab
        PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks')
        CACHE_DIR = Path("/content/cache")
    except ImportError:
        CACHE_DIR = PROJECT_ROOT / "cache" / "odir5k_ensemble_classification"

    DATASET_DIR = PROJECT_ROOT / "ODIR-5K"

    # Dataset configuration
    FILE_NAME = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
    TRAINING_SOURCE_PATH = 'ODIR-5K_Training_Images/'
    TESTING_SOURCE_PATH = 'ODIR-5K_Testing_Images/'
    LABEL_STRINGS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Abnormalities']
    VAL_FRACTION = 0.1

    # Image processing
    TARGET_SIZE = (230, 230)
    COLOR_MODE = 'rgb'
    COLOR_SHAPE_MAP = {'grayscale': (1,), 'rgb': (3,), 'rgba': (4,)}
    SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

    # Model configuration
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-5

    # Loss functions
    LOSS = {
        'output_odir5kmcc': 'categorical_crossentropy',
        'output_odir5kmlc': 'binary_crossentropy',
    }
    OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Metrics
    METRICS = {
        'output_odir5kmcc': [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(multi_label=True, name='auc')
        ],
        'output_odir5kmlc': [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(multi_label=True, name='auc')
        ],
    }

    # Model paths
    MODEL_DIR = PROJECT_ROOT / "Trained_Models" / "ODIR-5K-Ensemble-Classification"
    MODEL_SAVE_WEIGHTS = str(MODEL_DIR / 'ODIR5K_weights.weights.h5')
    MODEL_SAVE_FINAL = str(MODEL_DIR / 'ODIR5K_final.keras')
    CHECKPOINT_PATH = str(MODEL_DIR / 'ODIR5K.keras')

config = Config()

# %%
os.chdir(config.DATASET_DIR)
df = pd.read_excel(config.FILE_NAME)

# %%
left_eye_keywords = df['Left-Diagnostic Keywords'].copy().str.split("，")
right_eye_keywords = df['Right-Diagnostic Keywords'].copy().str.split("，")

test_df = df.copy()
LABEL_COLS = test_df.columns[7:]

def get_key_diagnosis_single(col_name):
    other_diag_cols = [col for col in LABEL_COLS if col != col_name]
    single_rows = test_df[(test_df[col_name] == 1) & (test_df[other_diag_cols].sum(axis=1) == 0)].index
    unique_keywords = set().union(*[set(left_eye_keywords[row]) | set(right_eye_keywords[row]) for row in single_rows])
    return list(unique_keywords)

all_key_single_label = [get_key_diagnosis_single(LABEL_COLS[i]) for i in range(len(config.LABEL_STRINGS))]

# %%
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, source_path, batch_size=32, img_size=(230, 230), color_mode='rgb', augment=False, task_name='joint', shuffle=True):
        self.df = df.reset_index(drop=True)
        self.source_path = source_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.task_name = task_name
        self.shuffle = shuffle
        self.n = len(self.df)
        self.on_epoch_end()

        self.augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(factor=0.1, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(factor=0.15, value_range=(0, 1)),
            tf.keras.layers.RandomContrast(factor=0.15),
        ])

    def __len__(self):
        return self.n // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(self.n)
        if self.shuffle: np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        images = []
        labels_mlc = []
        labels_mcc = []

        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.source_path, row['filename'])
            img = self.load_and_preprocess_image(img_path)
            images.append(img)

            # Label MLC (multi-label)
            mlc_label = row[config.LABEL_STRINGS].values.astype(np.float32)
            labels_mlc.append(mlc_label)

            # Label MCC (one-hot)
            # Pick the first active disease, or normal if none
            mcc_label = np.zeros(len(config.LABEL_STRINGS), dtype=np.float32)
            active_indices = np.where(mlc_label == 1)[0]
            if len(active_indices) > 0: mcc_label[active_indices[0]] = 1.0
            else: mcc_label[0] = 1.0 # Normal
            labels_mcc.append(mcc_label)

        X = np.array(images, dtype=np.float32)
        if self.augment: X = self.augmentation_layers(X, training=True).numpy()

        Y_mlc = np.array(labels_mlc)
        Y_mcc = np.array(labels_mcc)

        if self.task_name == 'multi_class_classification': return X, Y_mcc
        elif self.task_name == 'multi_label_classification': return X, Y_mlc
        elif self.task_name == 'joint': return X, {'output_odir5kmcc': Y_mcc, 'output_odir5kmlc': Y_mlc}

    def load_and_preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return np.zeros((*self.img_size, 3))

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop and Resize
        h, w = img.shape[:2]
        if w != h:
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img[top:top+side, left:left+side]

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img / 255.0

    def to_tf_dataset(self, name="train", augment=False):

        def gen_callable():
            # Iterate through each batch generator
            for i in range(len(self)):
                X, Y = self[i]
                # Unpack batch into individual samples
                for j in range(len(X)):
                    yield X[j], (Y['output_odir5kmcc'][j], Y['output_odir5kmlc'][j])

        # output signature (data types + shapes)
        output_signature = (
            tf.TensorSpec(shape=(*self.img_size, 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(8,), dtype=tf.float32),
                tf.TensorSpec(shape=(8,), dtype=tf.float32)
            )
        )

        ds = tf.data.Dataset.from_generator(gen_callable, output_signature=output_signature)

        cache_dir = config.CACHE_DIR / 'generator_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        for lockfile in cache_dir.glob(f"{name}*.lockfile"):
            try: os.remove(lockfile)
            except: pass

        ds = ds.cache(str(cache_dir / name))

        if name == "training": ds = ds.shuffle(buffer_size=min(self.n, 1000))
        ds = ds.batch(self.batch_size)

        def format_output(image, labels):
            if augment: image = self.augmentation_layers(image, training=True)
            return image, {'output_odir5kmcc': labels[0], 'output_odir5kmlc': labels[1]}

        ds = ds.map(format_output, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

class ODIR5kDataLoader:
    def __init__(self, df, all_key_single_label):
        self.df = df
        self.all_key_single_label = all_key_single_label
        self._prepare_data()

    def _prepare_data(self):
        records = []
        for _, row in self.df.iterrows():
            # Patient ID
            patient_id = str(row['ID'])

            # Left Eye
            left_img = row['Left-Fundus']
            left_keywords = row['Left-Diagnostic Keywords'].split('，')
            left_labels = self._keywords_to_label(left_keywords)
            records.append({
                'filename': left_img,
                'patient_id': patient_id,
                **{name: val for name, val in zip(config.LABEL_STRINGS, left_labels)}
            })

            # Right Eye
            right_img = row['Right-Fundus']
            right_keywords = row['Right-Diagnostic Keywords'].split('，')
            right_labels = self._keywords_to_label(right_keywords)
            records.append({
                'filename': right_img,
                'patient_id': patient_id,
                **{name: val for name, val in zip(config.LABEL_STRINGS, right_labels)}
            })

        self.expanded_df = pd.DataFrame(records)

    def _keywords_to_label(self, keywords):
        label = np.zeros(len(config.LABEL_STRINGS))
        for kw in keywords:
            kw_lower = kw.lower().strip()
            found = False
            for i, k_list in enumerate(self.all_key_single_label):
                if any(k.lower() == kw_lower for k in k_list):
                    label[i] = 1
                    found = True
            if not found and kw_lower != "":
                label[7] = 1 # Others/Abnormalities

        # If no specific disease found but labeled as normal... or if keywords list empty
        if np.sum(label) == 0:
            label[0] = 1

        return label

    def split_data(self, val_fraction=0.1):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
        train_idx, val_idx = next(gss.split(self.expanded_df, groups=self.expanded_df['patient_id']))

        train_df = self.expanded_df.iloc[train_idx]
        val_df = self.expanded_df.iloc[val_idx]

        return train_df, val_df

loader = ODIR5kDataLoader(df, all_key_single_label)
train_df, val_df = loader.split_data(val_fraction=config.VAL_FRACTION)

train_gen = DataGenerator(train_df, config.TRAINING_SOURCE_PATH, augment=False)
val_gen = DataGenerator(val_df, config.TRAINING_SOURCE_PATH, augment=False, shuffle=False)

train_ds = train_gen.to_tf_dataset(name="training", augment=True)
validation_ds = val_gen.to_tf_dataset(name="validation", augment=False)

# %%
model_odir5kmcc = tf.keras.models.load_model(odir5kmcc.CHECKPOINT_PATH)
model_odir5kmlc = tf.keras.models.load_model(odir5kmlc.CHECKPOINT_PATH)

backbone_odir5kmcc = tf.keras.Sequential(model_odir5kmcc.layers[:-4], name="backbone_odir5kmcc")
backbone_odir5kmlc = tf.keras.Sequential(model_odir5kmlc.layers[:-7], name="backbone_odir5kmlc")

inputs = tf.keras.layers.Input(shape=config.TARGET_SIZE + config.SHAPE_ADD)

feat_odir5kmcc = backbone_odir5kmcc(inputs)
feat_odir5kmcc = tf.keras.layers.GlobalAveragePooling2D()(feat_odir5kmcc)

feat_odir5kmlc = backbone_odir5kmlc(inputs)

feat_merged = tf.keras.layers.Concatenate()([feat_odir5kmcc, feat_odir5kmlc])

x = tf.keras.layers.Dense(512, activation='relu')(feat_merged)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

output_odir5kmcc = tf.keras.layers.Dense(8, activation='softmax', name='output_odir5kmcc')(x)
output_odir5kmlc = tf.keras.layers.Dense(8, activation='sigmoid', name='output_odir5kmlc')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=[output_odir5kmcc, output_odir5kmlc])

model.summary(line_length=100)
model.compile(
    loss=config.LOSS,
    optimizer=config.OPTIMIZER,
    metrics=config.METRICS,
    loss_weights={'output_odir5kmcc': 1.0, 'output_odir5kmlc': 1.0}
)

# %%
import gc
gc.collect()

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

weight_scheduler = TaskWeightScheduler(task_names=['output_odir5kmcc', 'output_odir5kmlc'])

callbacks = [
    weight_scheduler,
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(config.CHECKPOINT_PATH, monitor='val_output_odir5kmcc_accuracy', save_best_only=True, mode='max', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=config.EPOCHS,
    verbose=1,
    callbacks=callbacks
)

# %%
model.save_weights(config.MODEL_SAVE_WEIGHTS)
model.save(config.MODEL_SAVE_FINAL)

# %%
metrics = [
    ('loss', 'Total Loss'),
    ('output_odir5kmcc_loss', 'MCC Loss'),
    ('output_odir5kmlc_loss', 'MLC Loss'),
    ('output_odir5kmcc_accuracy', 'MCC Accuracy'),
    ('output_odir5kmlc_accuracy', 'MLC Accuracy'),
    ('output_odir5kmcc_auc', 'MCC AUC'),
    ('output_odir5kmlc_auc', 'MLC AUC'),
    ('output_odir5kmcc_precision', 'MCC Precision'),
    ('output_odir5kmlc_precision', 'MLC Precision'),
    ('output_odir5kmcc_recall', 'MCC Recall'),
    ('output_odir5kmlc_recall', 'MLC Recall'),
]
epochs = range(1, len(history.history['loss']) + 1)
for key, label in metrics:
    if key in history.history:
        plt.figure()
        plt.plot(epochs, history.history[key], 'r', label=f'Training {label}')
        if f'val_{key}' in history.history:
            plt.plot(epochs, history.history[f'val_{key}'], 'y', label=f'Validation {label}')
        plt.title(f'Training and validation {label}')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(label)
plt.show()

# %%
# Testing and Prediction on Test Set
test_files = sorted([f for f in os.listdir(config.TESTING_SOURCE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"\nTotal testing images found: {len(test_files)}")

def preprocess_test_image(img_path, target_size):
    img = cv2.imread(img_path)
    if img is None: return np.zeros((*target_size, 3))

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Square Crop
    h, w = img.shape[:2]
    side = min(w, h)
    img = img[(h-side)//2 : (h+side)//2, (w-side)//2 : (w+side)//2]

    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img / 255.0

# Table header for results
print(f"\n{'File Name':<25} | {'MCC Prediction':<20} | {'MLC Active Labels (Multi-label)':<40}")
print("-" * 90)

# Batch prediction for efficiency (optional, here doing one by one for clarity)
for i in range(min(100, len(test_files))): # Show first 100 predictions
    img_path = os.path.join(config.TESTING_SOURCE_PATH, test_files[i])
    img = preprocess_test_image(img_path, config.TARGET_SIZE)
    img_batch = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img_batch, verbose=0)
    # preds[0] -> output_odir5kmcc (Softmax - 8)
    # preds[1] -> output_odir5kmlc (Sigmoid - 8)

    # Process MCC (Single class)
    mcc_idx = np.argmax(preds[0][0])
    mcc_label = config.LABEL_STRINGS[mcc_idx]

    # Process MLC (Multiple labels)
    mlc_probs = preds[1][0]
    active_labels = [config.LABEL_STRINGS[j] for j, prob in enumerate(mlc_probs) if prob >= 0.5]
    mlc_label_str = ", ".join(active_labels) if active_labels else "None"

    print(f"{test_files[i]:<25} | {mcc_label:<20} | {mlc_label_str:<40}")
