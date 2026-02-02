import os
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

class EyeDiseasePredictor:
    LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    TARGET_SIZE = (512, 512)

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def crop_image(self, image):
        # image is a tf tensor
        mask = tf.reduce_sum(image, axis=-1) > 10
        non_zero_coords = tf.where(mask)

        if tf.shape(non_zero_coords)[0] == 0:
            return image

        y_min = tf.cast(tf.reduce_min(non_zero_coords[:, 0]), tf.int32)
        y_max = tf.cast(tf.reduce_max(non_zero_coords[:, 0]), tf.int32)
        x_min = tf.cast(tf.reduce_min(non_zero_coords[:, 1]), tf.int32)
        x_max = tf.cast(tf.reduce_max(non_zero_coords[:, 1]), tf.int32)

        image = tf.image.crop_to_bounding_box(image, y_min, x_min, y_max - y_min + 1, x_max - x_min + 1)
        return image

    def resize_image(self, image):
        image = tf.image.resize_with_pad(
            image, self.TARGET_SIZE[0],
            self.TARGET_SIZE[1],
            method=tf.image.ResizeMethod.BILINEAR
        )
        image.set_shape([self.TARGET_SIZE[0], self.TARGET_SIZE[1], 3])
        return image

    def clahe_cv2(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels + convert back to RGB
        lab = cv2.merge((l, a, b))
        image_res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image_res

    def apply_clahe(self, image):
        image = tf.cast(image, tf.uint8)
        image_shape = image.shape
        image = tf.numpy_function(func=self.clahe_cv2, inp=[image], Tout=tf.uint8)
        image.set_shape(image_shape)
        return image

    def preprocess(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image = self.crop_image(image)
        image = self.resize_image(image)
        image = self.apply_clahe(image)

        image = tf.cast(image, tf.float32)
        return tf.expand_dims(image, axis=0)

    def predict(self, img_path):
        img_batch = self.preprocess(img_path)
        preds = self.model.predict(img_batch, verbose=0)

        # mcc_preds = preds[0][0]  # Softmax - 8
        # mlc_preds = preds[1][0]  # Sigmoid - 8

        mcc_idx = np.argmax(preds[0][0])
        mcc_label = self.LABELS[mcc_idx]
        mcc_prob = float(preds[0][0][mcc_idx])

        mlc_probs = preds[1][0]
        mlc_results = {self.LABELS[j]: float(prob) for j, prob in enumerate(mlc_probs) if prob >= 0.5}

        return {
            "mcc_label": mcc_label,
            "mcc_prob": mcc_prob,
            "mlc_labels": mlc_results,
            "all_mlc_probs": {self.LABELS[j]: float(prob) for j, prob in enumerate(mlc_probs)}
        }
