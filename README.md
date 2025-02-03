Collecting workspace information

# Plant Disease Detection Project

## Overview

This project aims to develop a machine learning model to detect plant diseases from images of plant leaves. The project involves data preprocessing, model training, and evaluation to identify the best-performing model for plant disease classification.

## Project Structure

- **Data Preprocessing**: This step involves organizing the dataset into training, validation, and test sets. The images are resized, normalized, and augmented to improve the model's performance.

- **Model Training**: Various models are trained on the preprocessed data to identify the best-performing model. The models include Convolutional Neural Networks (CNN), VGG16, VGG19, ResNet, EfficientNet, and MobileNet.

- **Model Evaluation**: The trained models are evaluated on the validation set to determine their accuracy, precision, recall, F1 score, and confusion matrix. The best model is selected based on these metrics.

## Key Components

### Data Preprocessing

1. **Splitting the Data**: The dataset is split into training, validation, and test sets with a ratio of 70%, 15%, and 15%, respectively.
2. **Resizing Images**: All images are resized to a standard size of 224x224 pixels.
3. **Normalizing Images**: The pixel values of the images are normalized to a range of [0, 1].
4. **Data Augmentation**: Additional images are generated through transformations like rotation, width shift, height shift, shear, zoom, and horizontal flip to increase the diversity of the training set.

### Model Training

1. **CNN Model**: A simple Convolutional Neural Network with multiple layers to extract features from the images.
2. **Pre-trained Models**: Models like VGG16, VGG19, ResNet, EfficientNet, and MobileNet, which are pre-trained on large datasets, are fine-tuned for plant disease detection.

### Model Evaluation

1. **Metrics Calculation**: The models are evaluated based on accuracy, precision, recall, F1 score, and confusion matrix.
2. **Best Model Selection**: The model with the highest F1 score is selected as the best-performing model.

## Results

- **CNN Model**: Achieved an accuracy of 70.50%, precision of 0.71, recall of 0.70, and F1 score of 0.68.
- **MobileNet Model**: Achieved an accuracy of 89.28%, precision of 0.91, recall of 0.89, and F1 score of 0.89.

The MobileNet model outperformed the CNN model and was selected as the best model for plant disease detection.

## Conclusion

This project successfully developed a machine learning model to detect plant diseases from images. The MobileNet model demonstrated high accuracy and robustness, making it suitable for practical applications in agriculture to help farmers identify and manage plant diseases effectively.
