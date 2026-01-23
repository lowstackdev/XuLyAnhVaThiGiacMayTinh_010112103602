import os
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def draw_bounding_boxes(gradcam_path, output_dir, pred, labs, prob, img):
    if not os.path.exists(gradcam_path):
        return

    os.makedirs(output_dir, exist_ok=True)

    intensities = []

    img = cv2.imread(gradcam_path)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray) / 255.0   # normalized
    intensities.append(intensity)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        if box_w > 30 and box_h > 30:
            color = (0, 255, 0) if pred == "Normal" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+box_w, y+box_h), color, 2)

        if labs:
            lab = random.choice(labs)
            label_text = f"Detected: {pred}\nLikely: {lab} ({prob:.2f})"
            y0 = 30
            for i, line in enumerate(label_text.split("\n")):
                y = y0 + i * 20
                cv2.putText(img, line, (w - 350, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            label_text = f"{pred} ({prob:.2f})"
            cv2.putText(img, label_text, (w - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(gradcam_path))[0]}_highlighted.png")
        cv2.imwrite(save_path, img)
        print(f" Highlighted image saved at {save_path}")
