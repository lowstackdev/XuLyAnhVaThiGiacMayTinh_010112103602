import os
import cv2
import pandas as pd
import numpy as np

def create_video_from_images(img_path, output_video_path, fps=1):
    images = [img for img in sorted(os.listdir(img_path)) if img.endswith(".jpg") or img.endswith(".png")]

    if not images:
        print("No images found in {img_path}.")
        return

    first_image = cv2.imread(os.path.join(img_path, images[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_name in images:
        img_path = os.path.join(img_path, image_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (width, height))
            video.write(img)

    video.release()
    print(f" Video created successfully at: {output_video_path}")

def extract_frames_from_video(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"frame_{frame_count:04d}.png"
        frame_path = os.path.join(output_path, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from the video.")

# def blend_images(original, heatmap, alpha=0.6):
#     """Blend heatmap onto original image."""
#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     blended = cv2.addWeighted(heatmap_colored, alpha, original, 1 - alpha, 0)
#     return blended

# def load_heatmap(path, size):
#     heatmap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     heatmap = cv2.resize(heatmap, size)
#     heatmap = heatmap.astype(np.float32) / 255
#     return heatmap

# def annotate_image(image, label, confidence):
#     text = f"{label} ({confidence*100:.1f}%)"
#     color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
#     cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#     return image
