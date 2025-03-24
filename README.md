# DigitDetector-Neural-Network-for-Handwritten-Recognition

This repository contains two implementations of a neural network for classification tasks: a **Normal Code** and an **Optimized Code**. Both codes are designed to classify data from the `train.csv` file, which is assumed to contain image data (e.g., MNIST dataset). Below is an explanation of how each code works, their accuracy, and how they respond to new data. Additionally, a comparison of the two approaches is provided.

---

## **Normal Code**

### **Overview**
The normal code implements a simple feedforward neural network with one hidden layer. It uses the ReLU activation function for the hidden layer and softmax for the output layer. The network is trained using gradient descent with backpropagation.

### **Key Components**
1. **Data Loading and Preprocessing**:
   - The data is loaded from `train.csv` and shuffled to ensure randomness.
   - The dataset is split into a development set (`data_dev`) and a training set (`data_train`).

2. **Parameter Initialization**:
   - Weights (`w1`, `w2`) and biases (`b1`, `b2`) are initialized randomly.

3. **Forward Propagation**:
   - The input data is passed through the network to compute the output (`a2`) using ReLU and softmax activation functions.

4. **Backpropagation**:
   - Gradients are computed for weights and biases using the chain rule.
   - The gradients are used to update the parameters during training.

5. **Training**:
   - The model is trained using gradient descent for a fixed number of iterations.
   - Accuracy is printed every 50 iterations.

6. **Testing**:
   - A sample image from the training set is used to test the model's predictions.

### **Accuracy**
- The accuracy of the normal code is very low (~9.8%) and does not improve over iterations. This suggests that the model is not learning effectively.

### **Response to New Data**
- The model's poor accuracy indicates that it is not generalizing well to new data. It is likely underfitting due to insufficient complexity or improper hyperparameter tuning.

---

## **Optimized Code**

### **Overview**
The optimized code improves upon the normal code by introducing a more flexible architecture with multiple hidden layers. It also includes better initialization and numerical stability improvements.

### **Key Components**
1. **Data Loading and Preprocessing**:
   - Similar to the normal code, the data is loaded, shuffled, and split into development and training sets.

2. **Parameter Initialization**:
   - Parameters are initialized dynamically based on the specified layer sizes.
   - Weights are scaled by `0.01` to avoid large initial values, which can lead to instability.

3. **Forward Propagation**:
   - The network supports multiple hidden layers, with ReLU activation for hidden layers and softmax for the output layer.
   - A cache is maintained to store intermediate values for use in backpropagation.

4. **Backpropagation**:
   - Gradients are computed for all layers using the cached values from forward propagation.
   - The ReLU derivative is used to propagate errors backward.

5. **Training**:
   - The model is trained using gradient descent, with accuracy printed every 50 iterations.

6. **Testing**:
   - The model's predictions can be tested on new data, but this part is not explicitly shown in the provided code.

### **Accuracy**
- The accuracy of the optimized code is also low (~11.5%) and does not improve over iterations. This indicates that the model is still not learning effectively.

### **Response to New Data**
- Like the normal code, the optimized code does not generalize well to new data. The lack of improvement in accuracy suggests that further optimizations or architectural changes are needed.

---

## **Comparison: Normal vs Optimized Code**

| **Aspect**               | **Normal Code**                          | **Optimized Code**                       |
|--------------------------|------------------------------------------|------------------------------------------|
| **Architecture**          | Single hidden layer                      | Multiple hidden layers                   |
| **Parameter Initialization** | Random initialization with fixed sizes | Dynamic initialization based on layer sizes |
| **Forward Propagation**   | Fixed for one hidden layer               | Flexible for multiple layers             |
| **Backpropagation**       | Manual gradient computation              | Automated gradient computation           |
| **Accuracy**              | ~9.8% (no improvement)                  | ~11.5% (no improvement)                 |
| **Generalization**        | Poor                                    | Poor                                    |
| **Flexibility**           | Limited                                 | High                                    |

### **Which is Better?**
- The **optimized code** is better in terms of flexibility and scalability due to its support for multiple hidden layers and dynamic parameter initialization. However, both codes suffer from poor accuracy and generalization, indicating that further improvements are needed, such as:
  - Adjusting hyperparameters (learning rate, number of iterations).
  - Using more advanced optimization techniques (e.g., Adam optimizer).
  - Adding regularization (e.g., dropout, L2 regularization).
  - Increasing the complexity of the model (e.g., more layers, more neurons).

---

## **Training and Testing**

### **Training**
- Both codes use gradient descent to train the model on the training set (`x_train`, `y_train`).
- The training process involves:
  1. Forward propagation to compute predictions.
  2. Backpropagation to compute gradients.
  3. Updating parameters using the gradients.

### **Testing**
- The models are tested on a sample from the training set (`x_train`).
- The `test_predictions` function displays the model's prediction and the actual label for a given image.

### **Approach**
- The normal code uses a fixed architecture with one hidden layer.
- The optimized code allows for a flexible architecture with multiple hidden layers, making it more adaptable to different datasets and tasks.

---

![SVM Accuracy Graph](https://github.com/Ahnuf-Karim-Chowdhury/DigitDetector-Neural-Network-for-Handwritten-Recognition/blob/main/01.png?raw=true)
![SVM Accuracy Graph](https://github.com/Ahnuf-Karim-Chowdhury/DigitDetector-Neural-Network-for-Handwritten-Recognition/blob/main/06.png?raw=true)

## **Conclusion**
While the optimized code is more flexible and scalable, both implementations currently suffer from poor accuracy and generalization. To improve performance, consider experimenting with hyperparameters, adding regularization, or using more advanced optimization techniques. The optimized code provides a better foundation for further improvements due to its flexible architecture.
