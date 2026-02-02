# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from pathlib import Path

# %%
# %%
class Config:
    # Project paths
    try:
        import google.colab
        PROJECT_ROOT = Path('/content/drive/MyDrive/Colab Notebooks')
        TRAINING_PATH = "/content/training"
        VALIDATION_PATH = "/content/validation"
    except ImportError:
        if '__file__' in locals() or '__file__' in globals():
            PROJECT_ROOT = Path(__file__).parent.resolve()
        else:
            PROJECT_ROOT = Path(os.getcwd())

        TRAINING_PATH = "training"
        VALIDATION_PATH = "validation"

    DATASET_DIR = PROJECT_ROOT / "ODIR-5K"

    # Dataset configuration
    ANNOTATION_FILE_NAME = 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
    TRAINING_SOURCE_PATH = 'ODIR-5K_Training_Images'
    TESTING_SOURCE_PATH = 'ODIR-5K_Testing_Images'
    LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    VALIDATION_FRACTION = 0.1

    # Image processing
    TARGET_SIZE = (512, 512)
    COLOR_MODE = 'rgb'
    COLOR_SHAPE_MAP = {'grayscale': (1,), 'rgb': (3,), 'rgba': (4,)}
    SHAPE_ADD = COLOR_SHAPE_MAP.get(COLOR_MODE, (3,))

config = Config()

# %%
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def resize_image(image):
    image = tf.image.resize_with_pad(
        image, config.TARGET_SIZE[0],
        config.TARGET_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(image, tf.uint8)
    image.set_shape([config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3])
    return image

def crop_image(image):
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

def CLAHE(image):
    # uint8 format (0-255)
    image = tf.cast(image, tf.uint8)
    image_shape = image.shape

    # input numpy array
    image = tf.numpy_function(func=clahe_cv2, inp=[image], Tout=tf.uint8)

    # Reset shape
    image.set_shape(image_shape)
    return image

def clahe_cv2(image):
    # input numpy array
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

# %%
sample_image_path = config.DATASET_DIR / config.TRAINING_SOURCE_PATH / "0_left.jpg"

if not sample_image_path.exists():
    print(f"Image not found: {sample_image_path}")
else:
    # Load original image
    original_image = load_image(str(sample_image_path))

    # Apply pipeline steps
    cropped_image = crop_image(original_image)
    resized_image = resize_image(cropped_image)
    clahe_image = CLAHE(resized_image)

    # Convert tensors to numpy arrays for matplotlib
    original_np = original_image.numpy()
    cropped_np = cropped_image.numpy()
    resized_np = resized_image.numpy()
    clahe_np = clahe_image.numpy()

    # Create visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor='white')
    fig.suptitle('OPHTHALMIC IMAGE PROCESSING PIPELINE', fontsize=24, fontweight='bold', color='#2C3E50', y=0.98)

    titles = [
        '1. ORIGINAL IMAGE\n(ODIR-5K Raw Data)',
        '2. CROPPED IMAGE\n(Black Borders Removed)',
        '3. RESIZED IMAGE\n(512x512 with Padding)',
        '4. CLAHE ENHANCED\n(Contrast Optimized)'
    ]

    images = [original_np, cropped_np, resized_np, clahe_np]

    for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', color='#34495E', pad=15)
        ax.axis('off')

        # shape tag
        ax.text(0.02, 0.98, f'Shape: {img.shape}', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#3498DB", alpha=0.9),
                fontsize=11, color='#2980B9', verticalalignment='top')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the visualization
    save_path = config.PROJECT_ROOT / "pipeline_visualization.png"
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

    plt.show()

    # Print processing information
    print("Image Processing Pipeline Results:")
    print(f"Original image shape: {original_np.shape}")
    print(f"Cropped image shape: {cropped_np.shape}")
    print(f"Resized image shape: {resized_np.shape}")
    print(f"CLAHE enhanced image shape: {clahe_np.shape}")
    print(f"Sample image used: {sample_image_path.name}")
