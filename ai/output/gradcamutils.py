import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input

GRADCAM_SIZE = (1024, 1024)
MODEL_INPUT_SIZE = (224, 224)

def load_and_preprocess_image(img_path, target_size=MODEL_INPUT_SIZE):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
             return layer.name
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    raise ValueError("Could not find a convolutional layer in the model.")

def compute_gradcam_heatmap(model, img_array, target_class_index):
    last_conv_layer_name = find_last_conv_layer(model)

    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, target_class_index]

    grads = tape.gradient(loss, conv_outputs)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam
    cam = cv2.resize(cam, GRADCAM_SIZE)
    cam = np.uint8(255 * cam)

    return cam

def create_superimposed_image(heatmap, img_path, target_size=GRADCAM_SIZE):
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, target_size)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def save_image(img, path):
    cv2.imwrite(path, img)

def generate_gradcam(model, img_path, target_class_index, output_path):
    img_array = load_and_preprocess_image(img_path)
    cam = compute_gradcam_heatmap(model, img_array, target_class_index)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    superimposed_img = create_superimposed_image(heatmap, img_path)
    save_image(superimposed_img, output_path)
    return cam, superimposed_img
