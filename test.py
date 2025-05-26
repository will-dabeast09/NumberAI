
import math
import autograd.numpy as np
from autograd import grad
import tensorflow as tf

NUM_INPUTS = 28*28 # 784
NUM_HIDDEN_LAYERS = 2
SIZE_HIDDEN_LAYERS = 10
NUM_OUTPUTS = 10

LEARNING_RATE = 0.01
NUM_EPOCHS = 100
BATCH_SIZE = 64

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

def calculate_accuracy(y_true_orig_labels, y_pred_softmax):
    """
    Calculates accuracy.
    y_true_orig_labels: 1D array of original integer labels (e.g., [5, 0, 4,...])
    y_pred_softmax: 2D array of softmax probabilities from the network
    """
    predicted_labels = np.argmax(y_pred_softmax, axis=1)
    accuracy = np.mean(predicted_labels == y_true_orig_labels)
    return accuracy

# objective function for autograd
def objective_function(params, inputs, targets_one_hot):
    """
    Computes the cost for a given set of parameters and data.
    params: A list containing [weights_list, biases_list]
    """
    current_weights, current_biases = params[0], params[1]
    predictions = neural_network_predict(current_weights, current_biases, inputs)
    cost = cross_entropy_cost(targets_one_hot, predictions)
    return cost

# define functions to get gradients
compute_gradients = grad(objective_function, argnum=0)

def main():
    #MNIST
    print("Loading MNIST dataset...")
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()
    print("Dataset loaded.")

    x_train_flat = x_train_orig.reshape(x_train_orig.shape[0], NUM_INPUTS)
    x_test_flat = x_test_orig.reshape(x_test_orig.shape[0], NUM_INPUTS)

    x_train_normalized = x_train_flat.astype('float32') / 255.0
    x_test_normalized = x_test_flat.astype('float32') / 255.0

    y_train_one_hot = one_hot_encode(y_train_orig, NUM_OUTPUTS)
    y_test_one_hot = one_hot_encode(y_test_orig, NUM_OUTPUTS)

    #---------Initialization-----------
    print("\nInitializing network weights and biases...")
    global_weights = [] # Renamed to avoid conflict if main is called multiple times
    global_biases = []

    # Layer 0: Input to Hidden 1
    global_weights.append(np.random.uniform(-0.5, 0.5, (NUM_INPUTS, SIZE_HIDDEN_LAYERS)))
    global_biases.append(np.zeros(SIZE_HIDDEN_LAYERS))

    # Hidden layers (if NUM_HIDDEN_LAYERS > 1)
    for _ in range(NUM_HIDDEN_LAYERS - 1):
        global_weights.append(np.random.uniform(-0.5, 0.5, (SIZE_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS)))
        global_biases.append(np.zeros(SIZE_HIDDEN_LAYERS))

    # Last Hidden layer to Output
    global_weights.append(np.random.uniform(-0.5, 0.5, (SIZE_HIDDEN_LAYERS, NUM_OUTPUTS)))
    global_biases.append(np.zeros(NUM_OUTPUTS))
    print("Initialization complete.")
    #-----------------------------------

    print("\n--Starting Training--")
    num_samples_train = x_train_normalized.shape[0]
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        # shuffle training data at beginning of each epoch because it's good practice
        permutation = np.random.permutation(num_samples_train)
        shuffled_x_train = x_train_normalized[permutation]
        shuffled_y_train_one_hot = y_train_one_hot[permutation]

        for i in range(0, num_samples_train, BATCH_SIZE):
            batch_x = shuffled_x_train[i : i + BATCH_SIZE]
            batch_y_one_hot = shuffled_y_train_one_hot[i : i + BATCH_SIZE]

            # combine weights & biases into the params stucture expected by objective_function
            current_params = [global_weights, global_biases]

            # compute gradients!!!
            gradients = compute_gradients(current_params, batch_x, batch_y_one_hot)
            grad_weights = gradients[0]
            grad_biases = gradients[1]

            # update weights and biases using gradient descent
            for j in range(len(global_weights)):
                global_weights[j] -= LEARNING_RATE * grad_weights[j]
                global_biases[j] -= LEARNING_RATE * grad_biases[j]
    
    # Calculate loss and accuracy on the full training set (or a validation set)
    train_preds = neural_network_predict(global_weights, global_biases, x_train_normalized)
    train_loss = cross_entropy_cost(y_train_one_hot, train_preds)
    train_accuracy = calculate_accuracy(y_train_orig, train_preds)
            
    test_preds = neural_network_predict(global_weights, global_biases, x_test_normalized)
    test_loss = cross_entropy_cost(y_test_one_hot, test_preds)
    test_accuracy = calculate_accuracy(y_test_orig, test_preds)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy*100:.2f}%")

    final_test_preds = neural_network_predict(global_weights, global_biases, x_test_normalized)
    final_test_accuracy = calculate_accuracy(y_test_orig, final_test_preds)
    print(f"\nFinal Test Accuracy: {final_test_accuracy*100:.2f}%")


main()
