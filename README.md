# DigitDetector-Neural-Network-for-Handwritten-Recognition

This repository contains two implementations of a neural network for handwritten digit recognition using the MNIST dataset: a "normal" implementation and an "optimized" version. Both implementations aim to classify handwritten digits from the MNIST dataset using a simple neural network architecture.

## Dataset

The dataset used is the MNIST dataset, loaded from a CSV file ("train.csv"). Each row represents an image, with the first column being the label (digit 0-9) and the remaining columns being pixel values (0-255).

## Data Preprocessing

Both implementations start by loading the data, shuffling it, and splitting it into development (validation) and training sets.

-   **Shuffling:** The data is shuffled to ensure randomness in the training process.
-   **Splitting:**
    -      The first 1000 samples are used as the development set (`x_dev`, `y_dev`).
    -      The remaining samples (excluding the first 10,000 in the optimized and 1000 in the normal code) are used as the training set (`x_train`, `y_train`).
-   **Transposition:** The data is transposed so that each column represents an image, which is a common practice in neural network implementations for efficient matrix operations.

## Normal Code Explanation

### Architecture

The normal code uses a simple two-layer neural network.

-      **Initialization:** Weights (`w1`, `w2`) are initialized with small random values, and biases (`b1`, `b2`) are initialized with zeros.
-   **Forward Propagation:**
    -      Computes the weighted sum of inputs and biases (`z1`, `z2`).
    -      Applies the ReLU activation function to the first layer (`a1`).
    -   Applies the softmax activation function to the output layer (`a2`).
-   **Backpropagation:**
    -      Calculates the gradients of the loss function with respect to the weights and biases (`dw1`, `db1`, `dw2`, `db2`).
    -      Uses one-hot encoding for the target labels.
    -   Uses the derivative of the ReLU function.
-   **Parameter Updates:** Updates the weights and biases using gradient descent.
-   **Prediction:** Predicts the digit by taking the argmax of the output layer.
-   **Accuracy:** Calculates the accuracy of the predictions.
-   **Training:** The `gradient_descent` function trains the network for a specified number of iterations.
-   **Testing:** The `test_predictions` function displays a sample image and its predicted label.
-   **Test/Train:** The test set is the data_dev, and the train set is the data_train. The test set is used to validate the model's performance on unseen data, while the training set is used to adjust the model's parameters.

### Accuracy and Response to New Data

The normal code shows very poor accuracy in the provided output, consistently around 0.098. This indicates the model is not learning effectively. This poor accuracy may be due to a few reasons.
-   **Learning Rate:** The learning rate may be too high or too low.
-   **Initialization:** The weight initialization might not be optimal.
-   **Network Architecture:** The network architecture might be too simple.
-   **Training set size:** The normal code is training with a very large training set, this may be causing issues.
-   The code does have the ability to respond to new data by using the trained weights and biases to make predictions on the new data. However, due to its low accuracy, the predictions may be inaccurate.

## Optimized Code Explanation

### Architecture

The optimized code uses a more flexible multi-layer neural network architecture.

-   **Initialization:** Weights and biases are initialized with small random values using a more general initialization function that accepts layer sizes.
-   **Forward Propagation:**
    -      Uses a loop to handle multiple layers.
    -      Stores intermediate values in a cache for backpropagation.
    -   Uses ReLU activation for hidden layers and softmax for the output layer.
-   **Backpropagation:**
    -      Calculates gradients for all layers.
    -      Uses a loop to iterate through the layers in reverse order.
    -   Uses the cache to retrieve intermediate values.
-   **Parameter Updates:** Updates parameters using gradient descent and a learning rate.
-   **Prediction and Accuracy:** Similar to the normal code.
-   **Training:** The `gradient_descent` function trains the network for a specified number of iterations.
-   **Test/Train:** The test set is the data_dev, and the train set is the data_train. The test set is used to validate the model's performance on unseen data, while the training set is used to adjust the model's parameters.

### Accuracy and Response to New Data

The optimized code also shows poor accuracy, consistently around 0.11. This indicates similar learning issues as the normal code. However, it is slightly better than the normal code, indicating the changes made have a small positive effect. The optimized code has the ability to respond to new data by using the trained parameters to make predictions. However, due to its low accuracy, the predictions may be inaccurate.

## Comparison

-   **Flexibility:** The optimized code is more flexible due to its ability to handle multiple layers.
-   **Readability:** The optimized code is more organized and readable due to the use of dictionaries and loops.
-   **Accuracy:** Both implementations show poor accuracy, but the optimized code is slightly better.
-   **Efficiency:** The optimized code is more efficient due to the use of vectorized operations and optimized loops.
-   **Maintainability:** The optimized code is easier to maintain and extend due to its modular design.

## Test.ipynb Explanation

The `test.ipynb` file contains a slightly modified version of the "normal" code. The main purpose of this notebook is to test the model's predictions on sample images and visualize the results.

-   **Data Loading and Preprocessing:** Similar to the other implementations.
-   **Model Initialization, Forward Propagation, Backpropagation, and Parameter Updates:** Same as the normal code.
-   **Training:** The model is trained using gradient descent.
-   **Prediction and Visualization:** The `test_predictions` function displays a sample image and its predicted label.
-   **Test/Train:** The test set is the data_dev, and the train set is the data_train. The test set is used to validate the model's performance on unseen data, while the training set is used to adjust the model's parameters.

## Conclusion

The optimized code is better than the normal code due to its flexibility, readability, efficiency, and maintainability. However, both implementations require further tuning to improve their accuracy. The `test.ipynb` file provides a way to visualize the model's predictions and evaluate its performance.
