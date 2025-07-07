
#  Iris Dataset Classification using a Fully Connected Neural Network (PyTorch)

##  Overview

This project demonstrates how to implement a **Fully Connected (Dense) Neural Network** in **PyTorch** to classify samples from the famous **Iris dataset**. The network is built using basic principles of **Fully Connected Layers**, integrating activation functions, loss functions, and optimization techniques as covered in our deep learning learning path.

---

##  What We Covered

This project applies the following concepts:

- **Fully Connected (Dense) Layers**: Each neuron is connected to every neuron in the previous and next layer.
- **Activation Function (ReLU)**: Introduces non-linearity enabling the network to learn complex patterns.
- **CrossEntropy Loss**: Suitable for multi-class classification problems.
- **Adam Optimizer**: An efficient optimization algorithm.
- **Training & Testing Loop**: Standard procedure for forward pass, loss calculation, backward pass, and optimization.
- **Performance Evaluation**: Measuring classification accuracy on unseen test data.

---

##  Dependencies

Install the required Python libraries before running the code:

```bash
pip install torch torchvision scikit-learn

```


##  Dataset
The Iris dataset is a classic dataset for pattern recognition, containing:

- 150 samples

- 4 numerical features:

- Sepal length

- Sepal width

- Petal length

- Petal width

# - 3 classes:

- Setosa

- Versicolor

- Virginica

 ##  How to Run
1. Clone the repository or download the project files.

2. Navigate to the project directory.

3. Run the program:

``` bash

python iris_fc_net.py

```
## Expected Output
The training process will display the loss value every 20 epochs, followed by a final test accuracy result. Example:

```python
Epoch [20/100], Loss: 0.7431
...
Test Accuracy: 97.78%
```

##  Key Concepts Demonstrated
- Feature Integration and Abstraction through dense layers.

- Non-linearity introduction with activation functions.

- Decision making via output layers and loss functions.

- Overfitting control and regularization techniques (optional extensions).

-  Practical application of Universal Approximation Theorem by demonstrating how even simple networks can approximate complex functions.

## Learning Outcome
By completing this project, you will gain hands-on experience with:

-  Building Fully Connected Neural Networks from scratch in PyTorch.

- Structuring data pipelines for supervised classification.

-  Implementing activation functions, loss functions, and optimizers.

-  Evaluating model performance.

## License
This project is open-source and available under the MIT License.


Stephen Mungai | @MungaiMwangi001
