# TensorFlow Flower Image Classifier

This repository contains a complete implementation of an image classification project developed using TensorFlow and TensorFlow Hub as part of the Udacity "Intro to Machine Learning with TensorFlow" Nanodegree program.

## Overview

The project builds a deep learning model capable of classifying images of flowers into 102 distinct categories. The dataset used is the Oxford Flowers 102 dataset, and the model utilizes transfer learning with a pre-trained feature extractor from TensorFlow Hub. The final application allows for predicting flower types from new images using a trained model, with a fully functional command-line interface.

## Project Structure

├── predict.py            # Script to make predictions using the trained model  
├── model/                # Saved model directory  
├── label_map.json        # JSON file mapping category labels to flower names  
├── ImageClassifier.ipynb # Jupyter Notebook (for exploration)
├── README.md             # This file  

## Setup and Installation

To run this project, you need Python 3.7+ and the following dependencies:

- TensorFlow (>=2.x)  
- TensorFlow Datasets  
- TensorFlow Hub  
- NumPy  
- Matplotlib

Install the dependencies via pip:

```
pip install tensorflow tensorflow-datasets tensorflow-hub matplotlib
```

## Dataset

- Name: Oxford Flowers 102  
- Classes: 102  
- Source: TensorFlow Datasets (tfds.load('oxford_flowers102'))  
- Each image is labeled with one of 102 flower species.

## Training the Model

Use train.py to train your model:

```
python train.py --data_dir ./ --save_dir ./model --arch mobilenet_v2 --learning_rate 0.001 --epochs 5 --gpu
```

Arguments:

- --data_dir: Path to the dataset (TFDS handles loading).  
- --save_dir: Directory to save the trained model checkpoint.  
- --arch: Pre-trained model architecture (mobilenet_v2, efficientnet_b0, etc.).  
- --learning_rate: Learning rate for training.  
- --epochs: Number of epochs to train.  
- --gpu: Use GPU if available.

## Making Predictions

Use predict.py to predict the class of a given image:

```
python predict.py path_to_image model/checkpoint --top_k 5 --category_names label_map.json --gpu
```

Arguments:

- path_to_image: Path to an image file.  
- checkpoint: Trained model checkpoint directory.  
- --top_k: Number of top predictions to return.  
- --category_names: JSON file mapping class indices to names.  
- --gpu: Use GPU if available.

## Model Performance

The model achieves competitive accuracy using transfer learning, typically above 85% top-1 accuracy on the validation set after 5 epochs using mobilenet_v2.

## Usage Example

After training, you can predict using:

```
python predict.py flowers/test/image_06743.jpg model --top_k 3 --category_names label_map.json --gpu
```

Example output:

```
1. Sunflower - 95.23%  
2. Daffodil - 3.76%  
3. Black-eyed Susan - 0.79%
```

## License

This project is licensed under the MIT License.

## Credits

Developed as part of the Udacity "AI Programming with Python and TensorFlow" Nanodegree. 
