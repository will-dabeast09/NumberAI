import math
import random
# from PIL import Image # Pillow (PIL) for image loading - example, not used in direct translation

# --- Constants ---
CHANNELS = 3  # Typically 3 for RGB, 1 for grayscale
WIDTH = 25
HEIGHT = 25
INPUT_SIZE = WIDTH * HEIGHT  # Assuming grayscale or pre-processed single value per pixel
HIDDEN_LAYER_SIZE = 16
OUTPUTS = 10
NUM_HIDDEN_LAYERS = 2

# --- Neuron Class ---
class Neuron:
    def __init__(self, initial_weights=None): # Add initial_weights parameter, default to None
        self.value = 0.0
        if initial_weights is None:
            self.weights = [] # If no weights are provided, initialize as empty list
        else:
            # Ensure it's a list or similar iterable, and copy it to avoid modifying the original
            self.weights = list(initial_weights) 
        self.bias = 0.0 # It's good practice to have a bias, even if initialized to 0

# --- Activation and Helper Functions ---
def re_lu(input_val: float) -> float:
    """Rectified Linear Unit activation function."""
    return max(0.0, input_val)

def color_magnitude(r: int, g: int, b: int) -> float:
    """Calculates the normalized magnitude of a color vector."""
    # This function was in the C code but not used to process the input_layer.
    # It might be intended for preprocessing RGB pixels into single float values.
    return math.sqrt((r * r) + (g * g) + (b * b)) / (255 * math.sqrt(3.0))

def random_double(min_val: float, max_val: float) -> float:
    """Generates a random double between min_val and max_val."""
    return random.uniform(min_val, max_val)

def cost_function(output_layer_neurons: list[Neuron], desired_output_values: list[float]) -> float:
    """Calculates the sum of squared errors cost."""
    cost = 0.0
    if len(output_layer_neurons) != len(desired_output_values):
        raise ValueError("Output layer size and desired output size must match.")
    for i in range(len(output_layer_neurons)):
        diff = output_layer_neurons[i].value - desired_output_values[i]
        cost += diff * diff
    return cost

def softmax(output_layer_neurons: list[Neuron]) -> tuple[list[float], float]:
    """
    Calculates softmax probabilities for the output layer.
    Returns a list of probabilities and the sum of exponentials (for debugging).
    """
    num_outputs = len(output_layer_neurons)
    if num_outputs == 0:
        return [], 0.0

    raw_values = [neuron.value for neuron in output_layer_neurons]

    # Find the maximum value for numerical stability (prevents overflow)
    max_val = max(raw_values) if raw_values else 0.0

    # Calculate exponentials
    exp_values = [math.exp(val - max_val) for val in raw_values]

    # Sum of exponentials
    sum_exp_values = sum(exp_values)

    # Normalize to get probabilities
    if sum_exp_values == 0: # Avoid division by zero
        probabilities = [0.0] * num_outputs # Or [1.0 / num_outputs] * num_outputs for uniform
    else:
        probabilities = [val / sum_exp_values for val in exp_values]

    return probabilities, sum_exp_values

def main():
    random.seed()

    print(f"Initializing a dummy input layer of size {INPUT_SIZE} with random values.")
    input_layer_values = [random_double(0.0, 1.0) for _ in range(INPUT_SIZE)]

    layer_matrix = []
    output_layer = [Neuron() for _ in range(OUTPUTS)]

    print("Initializing weights and biases with random values (0 to 0.1)...")
    for layer in range(NUM_HIDDEN_LAYERS):
        # If this is the first hidden layer, the size of the weight array should be the number of neurons in the input layer
        if layer == 0:
            prev_layer_size = INPUT_SIZE
        # Otherwise, it should be the hidden layer size (the size of the previous layer)
        else:
            prev_layer_size = HIDDEN_LAYER_SIZE
        layer_matrix.append([Neuron([random_double(0, 0.1) for _ in range(prev_layer_size)]) for _ in range(HIDDEN_LAYER_SIZE)])

    # Output Layer
    for j in range(OUTPUTS):
        output_layer[j].bias = random_double(0, 0.1)
        output_layer[j].weights = [random_double(0, 0.1) for _ in range(HIDDEN_LAYER_SIZE)]

    print("Performing forward pass...")
    for layer in range(NUM_HIDDEN_LAYERS):
        # If this is the first hidden layer, the size of the weight array should be the number of neurons in the input layer
        if layer == 0:
            prev_layer_size = INPUT_SIZE
        # Otherwise, it should be the hidden layer size (the size of the previous layer)
        else:
            prev_layer_size = HIDDEN_LAYER_SIZE
        
        for nidx in range(HIDDEN_LAYER_SIZE):
            neuron = layer_matrix[layer][nidx]
            activation = neuron.bias
            for j in range(prev_layer_size):
                activation += input_layer_values[j] * neuron.weights[j]
            neuron.value = re_lu(activation)

    # Calculate the output layer (raw logits)
    for j in range(OUTPUTS):
        neuron = output_layer[j]
        activation = neuron.bias
        for k in range(HIDDEN_LAYER_SIZE):
            activation += layer_matrix[NUM_HIDDEN_LAYERS - 1][k].value * neuron.weights[k]
        neuron.value = activation

    for layer in range(NUM_HIDDEN_LAYERS):
        if layer == 0:
            prev_layer_size = INPUT_SIZE
        # Otherwise, it should be the hidden layer size (the size of the previous layer)
        else:
            prev_layer_size = HIDDEN_LAYER_SIZE
        for nidx in range(HIDDEN_LAYER_SIZE):
            for weight in range(prev_layer_size):
                print(layer_matrix[layer][nidx].weights[weight], nidx, weight)

    for nidx in range(OUTPUTS):
        for weight in range(HIDDEN_LAYER_SIZE):
            print(output_layer[nidx].weights[weight])

    # --- Output Information ---
    desired_output_values = [0.0] * OUTPUTS
    if OUTPUTS > 0:
        desired_output_values[0] = 1.0  # Placeholder: "desired" class is the first one

    print("\nDesired Output:")
    print(*(f"{val:.6f}" for val in desired_output_values))

    print("\nRaw Output Values (Logits):")
    print(*(f"{neuron.value:.6f}" for neuron in output_layer))

    softmax_probabilities, _ = softmax(output_layer)
    print("\nSoftmax Probabilities:")
    print(*(f"{prob:.6f}" for prob in softmax_probabilities))

    cost = cost_function(output_layer, desired_output_values)
    print(f"\nCost (Sum of Squared Errors on Raw Outputs): {cost:.6f}\n")

if __name__ == "__main__":
    main()