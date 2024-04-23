# Brain Tumor Classification using MRI Images

## Overview
This project focuses on classifying brain tumor images using deep learning techniques. It utilizes MRI images to accurately classify tumors into different categories: glioma tumor, meningioma tumor, pituitary tumor, or no tumor. The model is built using Convolutional Neural Networks (CNNs) and trained on a dataset containing MRI images of various brain tumors.

## Dataset
The dataset consists of MRI images categorized into four classes: glioma tumor, meningioma tumor, pituitary tumor, and images with no tumor. The images are preprocessed and resized to ensure compatibility with the CNN model.

## Model Architecture
The CNN model architecture comprises several convolutional layers followed by max-pooling layers and dropout layers to prevent overfitting. The final layer utilizes softmax activation to classify the input image into one of the four tumor categories.

## Training
The model is trained using the Adam optimizer with categorical cross-entropy loss. Training is conducted over multiple epochs, with validation performed on a separate subset of the dataset to monitor model performance and prevent overfitting.

## Evaluation
The trained model's performance is evaluated using accuracy metrics on both the training and validation sets. Additionally, loss curves are plotted to visualize the model's convergence during training.

## Deployment
The trained model can be deployed to predict tumor categories for new MRI images. The deployment process involves loading the saved model and using it to classify new images.

## Predictions
The model can predict the tumor category for new MRI images. Users can input an MRI image, and the model will output the predicted tumor category.

## Created By
Abdul Mannan

---

*Note: The model's performance can vary based on the quality and diversity of the dataset used for training. Further improvements and fine-tuning may be necessary for optimal performance in real-world applications.*

