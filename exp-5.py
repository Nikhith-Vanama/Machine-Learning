import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network structure
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weights for input to hidden layer and hidden to output layer
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.rand(self.hidden_size)
        
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.random.rand(self.output_size)

    def forward(self, X):
        # Forward propagation through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backward(self, X, y):
        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs=1000):
        # Training the network over a specified number of epochs
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        # Predict function for testing
        output = self.forward(X)
        return np.argmax(output, axis=1)


# Load and prepare the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the neural network
input_size = X.shape[1]    # Number of features in the dataset (4 for Iris)
hidden_size = 5            # Number of neurons in the hidden layer
output_size = y_encoded.shape[1]  # Number of output classes (3 for Iris)

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
nn.train(X_train, y_train, epochs=1000)

# Make predictions on the test set
y_pred = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels back to integers

# Calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f'\nTest Accuracy: {accuracy * 100:.2f}%')