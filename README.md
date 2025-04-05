# Fashion MNIST Classification using Convolutional Neural Network (CNN)

## Project Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset is a collection of 28x28 grayscale images of clothing items, such as T-shirts, trousers, and shoes. The goal of the project is to train a CNN model to accurately predict the class label of each image.

## Project Structure

The project is structured as follows:

1. **Data Loading and Preprocessing:**
   - The Fashion MNIST dataset is loaded using `tf.keras.datasets.fashion_mnist.load_data()`.
   - The pixel values are normalized to the range [0, 1] to improve model performance.
   - A channel dimension is added to the images to make them compatible with the CNN input shape.
   - The labels are one-hot encoded using `tf.keras.utils.to_categorical()`.

2. **Model Building:**
   - A CNN model is constructed using the `tensorflow.keras` library.
   - The model consists of multiple convolutional blocks, each with convolutional layers, batch normalization, max pooling, and dropout for regularization.
   - Global average pooling is used to reduce the spatial dimensions of the feature maps.
   - Fully connected layers are added to perform the final classification.

3. **Model Compilation and Training:**
   - The model is compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.
   - Early stopping is implemented to prevent overfitting.
   - The model is trained on the training data with a validation split of 0.2.

4. **Model Evaluation and Visualization:**
   - The model's performance is evaluated on the test data using `model.evaluate()`.
   - Training and validation accuracy and loss are plotted using `matplotlib.pyplot`.
   - Predictions on sample test images are visualized using `seaborn` and `matplotlib.pyplot`.

## Logic and Algorithms

- **Convolutional Neural Networks (CNNs):** CNNs are a type of deep learning model specifically designed for image recognition tasks. They use convolutional layers to extract features from images and learn spatial hierarchies of patterns.

- **Batch Normalization:** Batch normalization helps stabilize and accelerate the training process by normalizing the activations of each layer.

- **Max Pooling:** Max pooling reduces the spatial dimensions of the feature maps, which helps to reduce the number of parameters and prevent overfitting.

- **Dropout:** Dropout is a regularization technique that randomly drops out neurons during training, which helps to prevent overfitting.

- **Global Average Pooling:** Global average pooling computes the average value of each feature map, which reduces the spatial dimensions to a single value per feature map.

- **Fully Connected Layers:** Fully connected layers perform the final classification by mapping the learned features to the output classes.

- **Adam Optimizer:** Adam is an optimization algorithm that is widely used for training deep learning models.

- **Categorical Cross-Entropy Loss:** Categorical cross-entropy is a loss function that is used for multi-class classification problems.

- **Accuracy Metric:** Accuracy is a metric that measures the percentage of correctly classified images.

## Technology Used

- **Python:** The primary programming language used for this project.
- **TensorFlow:** A popular deep learning framework used for building and training the CNN model.
- **Keras:** A high-level API for TensorFlow that simplifies the model building process.
- **Matplotlib:** A library for creating static, interactive, and animated visualizations in Python.
- **Seaborn:** A library for creating statistical graphics in Python.

![image](https://github.com/user-attachments/assets/2ddf095d-bda9-4859-af5d-5e75d52386e1)


## Conclusion

This project successfully demonstrates the application of a CNN for image classification using the Fashion MNIST dataset. The model achieves high accuracy on the test data, indicating its effectiveness in recognizing clothing items. The project also showcases the use of various deep learning techniques and libraries for building, training, and evaluating CNN models.
