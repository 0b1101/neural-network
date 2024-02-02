import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# Activation function - Sigmoid
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid function
def sigmoid_derivative(x: np.ndarray):
    return x * (1 - x)

# Mean Squared Error Loss
def mse_loss(y: np.ndarray, y_hat: np.ndarray):
    return np.mean(np.power(y - y_hat, 2))

# Derivative of Mean Squared Error Loss
def mse_loss_derivative(y: np.ndarray, y_hat: np.ndarray):
    return y_hat - y

# Z-Score Normalization
def normalization(val):
    mean = np.mean(val, axis=0)
    std = np.std(val, axis=0)

    normalized_val = (val - mean) / std

    return normalized_val

# Splitting dataset into train and test sets
def dataset_split(x, y, random_state = None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    test_size = int(0.2 * len(x))
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    return x_train, x_test, y_train, y_test

# Create batches for training
def create_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        batches = [(x[i: i + batch_size], y[i: i + batch_size])]

    return batches

# Neural Network Layer class
class Layer:
    def __init__(self, input_dim: int, output_dim: int, alpha = 0.01):
        self.weights = 2 * np.random.random((input_dim, output_dim)) - 1
        self.biases = np.zeros((1, output_dim))
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None
        self.alpha = alpha

    def forward(self, input_data) -> np.ndarray:
        self.input = input_data
        self.output = sigmoid(np.dot(input_data, self.weights) + self.biases)

        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        delta = output_error * sigmoid_derivative(self.output)
        input_error = np.dot(delta, self.weights.T)
        weight_error = np.dot(self.input.T, delta)

        self.weights -= learning_rate * (weight_error + self.alpha * self.weights)
        self.biases -= learning_rate * np.sum(delta, axis = 0, keepdims = True)

        return input_error

# Forward pass through the neural network
def forward(input: np.ndarray, layers: list[Layer]):
    current_input = input

    for layer in layers:
        current_input = layer.forward(current_input)
    return current_input

# Backward pass through the neural network
def backward(y: np.ndarray, y_hat: np.ndarray, layers: list[Layer], learning_rate: float) -> None:
    output_error = mse_loss_derivative(y, y_hat)

    for layer in reversed(layers):
        output_error = layer.backward(output_error, learning_rate)

# Calculate total loss with regularization term
def total_loss(y, y_hat, layers):
    mse = mse_loss(y, y_hat)

    regularization_term = sum(np.sum(layer.weights**2) for layer in layers)

    return mse + 0.5 * layers[0].alpha * regularization_term

# Calculate accuracy and confusion matrix
def calculate_accuracy(y_true, y_pred):
    y_pred_binary = np.round(y_pred)  # round predictions to binary (0 or 1)

    accuracy = accuracy_score(y_true, y_pred_binary)
    cmatrix = confusion_matrix(y_true, y_pred_binary)

    return accuracy, cmatrix

# K-Fold Cross-Validation
def kfold_cv(x, y, k, alphas, learning_rates, hidden_layers, epochs, batch_size):
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    fold_size = len(x) // k
    average_losses = {}

    for alpha in alphas:
        for lr in learning_rates:
            print(f"\nValidating for alpha: {alpha}, learning rate: {lr}")
            learning_rate = lr
            losses = []

            for i in range(k):
                val_indices = indices[i * fold_size: (i + 1) * fold_size]
                train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

                x_train, x_test = x[train_indices], x[val_indices]
                y_train, y_test = y[train_indices], y[val_indices]

                layers = [Layer(x_train.shape[1], hidden_layers[0], alpha)]
                for i in range(1, len(hidden_layers)):
                    layers.append(Layer(hidden_layers[i-1], hidden_layers[i], alpha))
                layers.append(Layer(hidden_layers[-1], y_train.shape[1], alpha))

                for epoch in range(epochs):
                    i = 0
                    for x_batch, y_batch in create_batches(x_train, y_train, batch_size):
                        y_hat = forward(x_batch, layers)
                        backward(y_batch, y_hat, layers, learning_rate)

                    learning_rate /= (1 + i)

                y_hat_val = forward(x_test, layers)
                loss = total_loss(x_test, y_hat_val, layers)
                losses.append(loss)

            average_losses[(alpha, lr)] = np.mean(losses)
            print(f"\r  Average Loss: {average_losses[(alpha, lr)]}          ")

    return average_losses

# Hyperparameter testing using K-Fold Cross-Validation
def hiperparameters_test(x, y):
    k = 5
    alphas = [0.001, 0.01, 0.1, 0.5, 1]
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1]

    epochs = 1000
    batch_size = 32
    hidden_layers = [2, 2]

    average_losses = kfold_cv(x, y, k, alphas, learning_rates, hidden_layers, epochs, batch_size)

    best_params = min(average_losses, key = average_losses.get)
    lr, alpha = best_params
    print(f"\nBest hiperparameters: Learning rate = {lr}, Regularization term = {alpha}\n")

    # Plot the results
    plot_heatmap_hyperparameters(learning_rates, alphas, average_losses)

    return best_params

def plot_heatmap_hyperparameters(learning_rates, alphas, average_losses):
    accuracy_values_2d = np.array([[average_losses[(lr, alpha)] for alpha in alphas] for lr in learning_rates])

    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_values_2d, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True,
                xticklabels=np.unique(alphas), yticklabels=np.unique(learning_rates))

    plt.xlabel('Alpha')
    plt.ylabel('Learning Rate')
    plt.title('Heat map of the average loss for various hiperparameters')
    plt.savefig('readme-images/heatmap.png') 

def main():
    data = load_breast_cancer()
    x = data.data
    y = data.target.reshape(-1, 1)

    # Normalize features using Z-Score
    x = normalization(x)

    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = dataset_split(x, y, 42)

    # Hyperparameters
    params = hiperparameters_test(x_train, y_train)
    learning_rate, alpha = params

    epochs = 10000
    batch_size = 32
    hidden_layers = [2, 2]

    # Initialize layers with regularization term
    layers = [Layer(x_train.shape[1], hidden_layers[0], alpha)]

    for i in range(len(hidden_layers) - 1):
        layers.append(Layer(hidden_layers[i], hidden_layers[i + 1], alpha))
    layers.append(Layer(hidden_layers[-1], y_train.shape[1], alpha))

    batches = create_batches(x_train, y_train, batch_size)

    # Train the model with batches
    for epoch in range(epochs):
        i = 0

        for batch_x, batch_y in batches:
            # Forward pass
            y_hat = forward(batch_x, layers)

            # Loss with regularization term
            loss = total_loss(batch_y, y_hat, layers)

            # Backward
            backward(batch_y, y_hat, layers, learning_rate)

        # Adjust the learning rate
        learning_rate /= (1 + i)

        if epoch % 1000 == 0:
          print(f"\n\rEpoch {epoch} Loss: {np.mean(loss)}")

    y_hat_test = forward(x_test, layers)

    # Calculate and print accuracy
    accuracy, cmatrix = calculate_accuracy(y_test, y_hat_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = [False, True])

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    cm_display.plot()
    plt.savefig('readme-images/confusion-matrix.png') 
    
if __name__ == "__main__":
    main()
