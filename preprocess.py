import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to 128x128 pixels
    image = image / 255.0  # Normalize pixel values
    return image
