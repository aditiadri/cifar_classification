
# Deep Learning Project with CIFAR-10 Dataset

## Overview
This repository contains a deep learning project using the CIFAR-10 dataset. The project includes loading the dataset, building a convolutional neural network (CNN) model, training the model, and evaluating its performance on the test set. Additionally, data augmentation techniques are explored to enhance model generalization.

## Prerequisites
- Python 3
- Libraries: pandas, numpy, matplotlib, seaborn, keras, tensorflow
- Jupyter Notebook (optional)

## Dataset
The CIFAR-10 dataset is used, consisting of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.


## Instructions
1. Install the required libraries: `pip install pandas numpy matplotlib seaborn keras tensorflow`.
2. Run `load_data.ipynb` to load and explore the CIFAR-10 dataset.
3. Run `build_train_model.ipynb` to build and train the CNN model.
4. Optionally, run `data_augmentation.ipynb` to explore data augmentation techniques.

## Model
The CNN model consists of multiple convolutional and dense layers. The model is compiled using categorical crossentropy loss and RMSprop optimizer.

## Training
The model is trained on the training set with a specified number of epochs and batch size.

## Evaluation
The trained model is evaluated on the test set, and the accuracy is reported.

## Data Augmentation
Data augmentation techniques such as rotation, width shift, horizontal flip, and vertical flip are explored to improve model generalization.

## Saved Models
Trained models are saved in the `saved_models/` directory. 

## Future Work
- Experiment with different hyperparameters.
- Explore additional data augmentation techniques.
- Fine-tune the model for improved performance.

Feel free to contribute and provide feedback!

