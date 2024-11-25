import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import os
from functools import partial

# Create a directory for results if it doesn't exist
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Multi-layer Perceptron class definition
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation='tanh'):
        np.random.seed(0)
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

    def forward_pass(self, inputs):
        self.inputs = inputs
        # First layer transformation
        self.hidden_linear = inputs @ self.weights_input_hidden + self.bias_hidden

        # Apply activation function
        if self.activation_type == 'tanh':
            self.hidden_activations = np.tanh(self.hidden_linear)
        elif self.activation_type == 'relu':
            self.hidden_activations = np.maximum(0, self.hidden_linear)
        elif self.activation_type == 'sigmoid':
            self.hidden_activations = 1 / (1 + np.exp(-self.hidden_linear))
        else:
            raise ValueError('Invalid activation function specified')

        # Output layer
        self.output_linear = self.hidden_activations @ self.weights_hidden_output + self.bias_output
        self.predictions = 1 / (1 + np.exp(-self.output_linear))
        return self.predictions

    def backpropagation(self, inputs, targets):
        num_samples = targets.shape[0]
        error_output_layer = self.predictions - targets
        gradient_weights_output = (self.hidden_activations.T @ error_output_layer) / num_samples
        gradient_bias_output = np.sum(error_output_layer, axis=0, keepdims=True) / num_samples

        error_hidden_layer = error_output_layer @ self.weights_hidden_output.T
        if self.activation_type == 'tanh':
            gradient_hidden_layer = error_hidden_layer * (1 - np.tanh(self.hidden_linear) ** 2)
        elif self.activation_type == 'relu':
            gradient_hidden_layer = error_hidden_layer * (self.hidden_linear > 0).astype(float)
        elif self.activation_type == 'sigmoid':
            sigmoid_values = 1 / (1 + np.exp(-self.hidden_linear))
            gradient_hidden_layer = error_hidden_layer * sigmoid_values * (1 - sigmoid_values)
        else:
            raise ValueError('Invalid activation function specified')

        gradient_weights_input = (inputs.T @ gradient_hidden_layer) / num_samples
        gradient_bias_hidden = np.sum(gradient_hidden_layer, axis=0, keepdims=True) / num_samples

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * gradient_weights_output
        self.bias_output -= self.learning_rate * gradient_bias_output
        self.weights_input_hidden -= self.learning_rate * gradient_weights_input
        self.bias_hidden -= self.learning_rate * gradient_bias_hidden

        # Store gradients for visualization
        self.gradient_weights_input = gradient_weights_input
        self.gradient_weights_output = gradient_weights_output

# Data generation for training
def create_dataset(samples=100):
    np.random.seed(0)
    data = np.random.randn(samples, 2)
    labels = (data[:, 0] ** 2 + data[:, 1] ** 2 > 1).astype(int) * 2 - 1
    return data, labels.reshape(-1, 1)

# Function to handle visualization updates
def update_visualization(frame_idx, network, ax_input, ax_hidden, ax_gradient, data, labels, *limits):
    # Clean up previous drawings
    ax_input.clear()
    ax_hidden.clear()
    ax_gradient.clear()

    # Train network with multiple steps
    for _ in range(10):
        network.forward_pass(data)
        network.backpropagation(data, labels)

    step = frame_idx * 10

    # Visualization for input and gradients
    ax_input.set_title(f'Input Space at Step {step}')
    ax_gradient.set_title(f'Gradients at Step {step}')
    ax_hidden.set_title(f'Hidden Layer Space at Step {step}')

# Main function for configuration and execution
if __name__ == "__main__":
    act_fn = "tanh"  # Options: 'relu', 'sigmoid'
    learning_rate = 0.1
    steps = 1000
    # Assume `visualize` function is defined elsewhere
    visualize(act_fn, learning_rate, steps)
