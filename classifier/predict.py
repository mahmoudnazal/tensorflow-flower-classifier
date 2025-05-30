import argparse
import tensorflow as tf
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras import models
from PIL import Image

def process_image(image_path):
    # Preprocess image for model prediction
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    return image

def predict(image_path, model_path, top_k, category_names):
    # Load image and model, perform prediction, and display top results
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)
    model = models.load_model(model_path, compile=False)
    predictions = model.predict(image)[0]
    top_k_indices = predictions.argsort()[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    top_k_labels = [str(i + 1) for i in top_k_indices]

    if category_names:
        with open(category_names, 'r') as f:
            class_names = json.load(f)
        top_k_names = [class_names.get(label, label) for label in top_k_labels]
    else:
        top_k_names = top_k_labels

    for name, prob in zip(top_k_names, top_k_probs):
        print(f"{name}: {prob:.4f}")
