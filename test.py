
import math
import autograd.numpy as np
from autograd import grad
import tensorflow as tf

NUM_INPUTS = 28*28 # 784
NUM_HIDDEN_LAYERS = 2
SIZE_HIDDEN_LAYERS = 10
NUM_OUTPUTS = 10

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def mse_cost(n, obs, pred):
    return np.sum((obs - pred)**2) / n

def neural_network_predict(weights, biases, inputs):
    # weights (list) - a list of NumPy arrays, where each array is the weight matrix for a layer
    # biases (list) - a list of Numpy arrays, where each array is the bias vector for a layer
    # inputs (np.array) - the input data for the network - shape: (num_samples, NUM_INPUTS)
    current_layer_activation = inputs
    for i in range(NUM_HIDDEN_LAYERS):
        lin_out = (current_layer_activation @ weights[i]) + biases[i]
        current_layer_activation = sigmoid(lin_out)
    
    fin_lin_out = current_layer_activation @ weights[NUM_HIDDEN_LAYERS] + biases[NUM_HIDDEN_LAYERS]
    final_predictions = softmax(fin_lin_out)

    return final_predictions

def cross_entropy_cost(y_true_one_hot, y_pred_softmax):
    # y_true_one_hot: (num_samples, NUM_OUTPUTS), one-hot encoded true labels
    # y_pred_softmax: (num_samples, NUM_OUTPUTS), softmax probabilities
    # Add a small epsilon to prevent log(0)
    epsilon = 1e-12
    y_pred_softmax = np.clip(y_pred_softmax, epsilon, 1. - epsilon)
    log_likelihood = -np.sum(y_true_one_hot * np.log(y_pred_softmax), axis=1)
    return np.mean(log_likelihood)

def one_hot_encode(labels, num_classes):
    """Converts integer labels to one-hot encoded format using autograd.numpy."""
    # `labels` should be a 1D array of integers
    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return encoded

def main():
    #---------Initialization-----------
    weights = []
    biases = []

    # weights & biases from input layer to first hidden layer
    weights.append(np.random.uniform(-0.5, 0.5, (NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS)))
    biases.append(np.zeros(SIZE_HIDDEN_LAYERS))

    # weights & biases between hidden layers
    for _ in range(NUM_HIDDEN_LAYERS - 1):
        weights.append(np.random.uniform(-0.5, 0.5, (SIZE_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS)))
        biases.append(np.zeros(SIZE_HIDDEN_LAYERS))
    
    # weights and biases from last hidden layer to output layer
    weights.append(np.random.uniform(-0.5, 0.5, (SIZE_HIDDEN_LAYERS, NUM_OUTPUTS)))
    biases.append(np.zeros(NUM_OUTPUTS))
    #-----------------------------------

    print("Loading MNIST dataset...")
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()
    print("Dataset loaded.")

    x_train_flat = x_train_orig.reshape(x_train_orig.shape[0], NUM_INPUTS)
    x_test_flat = x_test_orig.reshape(x_test_orig.shape[0], NUM_INPUTS)

    x_train_normalized = x_train_flat.astype('float32') / 255.0
    x_test_normalized = x_test_flat.astype('float32') / 255.0

    y_train_one_hot = one_hot_encode(y_train_orig, NUM_OUTPUTS)
    y_test_one_hot = one_hot_encode(y_test_orig, NUM_OUTPUTS)

    

main()
