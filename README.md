# Transfer Learning for Video Classification

## Introduction

This project aims to build a classifier capable of distinguishing videos of five different activities using Transfer Learning. The approach involves training a Convolutional Neural Network (CNN) for image classification and transforming it into a video classifier using moving averages. The project is implemented in Keras and Python.

## Video Classification Approach

Videos are treated as sequences of individual images, and the task is approached as image classification for each frame. To capture the temporal nature of videos, a combination of Recurrent Neural Network (RNN) and CNN is usually used. However, this project simplifies the process using a Moving Averages over Predictions approach.

### Moving Averages over Predictions for Video Classification

Instead of directly training an RNN-CNN, a CNN is trained for image classification. The video classifier is then created by taking moving averages of predictions. This helps in mitigating issues like flickering, where different labels may be assigned to different frames of the same video.

## Data Exploration and Pre-processing

1. Images for each class are organized in separate folders within the "Sport Images" directory.
2. Randomly select a portion of images for training, validation, and testing.
3. Ensure uniform image sizes by zero-padding or resizing, using tools like OpenCV.

## Transfer Learning for Image Classification

1. Utilize pre-trained models ResNet50, EfficientNetB0, and VGG16.
2. Freeze all layers except the last fully connected layer and use the outputs of the penultimate layer as features.
3. Apply empirical regularization by augmenting images in the training set through cropping, random zooming, rotation, flipping, contrast adjustment, and translation.
4. Use ReLU activation functions, softmax layer, batch normalization, 20% dropout rate, ADAM optimizer, and multinomial cross-entropy loss.
5. Train the networks for at least 50 epochs, preferably 100, with early stopping using the validation set.

### Evaluation Metrics

- Plot training and validation errors vs. epochs.
- Report Confusion Matrix, Precision, Recall, Accuracy, and F1 score for both training and test sets.

## Video Classification Using Moving Averages

1. Reuse validation and test data for further training without overfitting.
2. Apply at least L equally spaced frames (L >= 100) from each video in the "Sport Videos" folder.
3. Calculate the average probability vectors for each video.
4. Select the class with the maximum probability in the vector for each video and compare it to the actual label.

### Evaluation Metrics

- Report Confusion Matrix, Precision, Recall, Accuracy, and F1 score for the video classification on the test data.

## Conclusion

This README provides an overview of the project, outlining the approach, data handling, transfer learning strategy, and evaluation metrics. For detailed implementation, please refer to the project code and associated resources.
