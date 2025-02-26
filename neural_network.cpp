#include "neural_network.h"
#include <cmath>
#include <random>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    
    // Initialize weights with random values
    weights_input_hidden = std::vector<std::vector<double>>(input_size, std::vector<double>(hidden_size));
    weights_hidden_output = std::vector<std::vector<double>>(hidden_size, std::vector<double>(output_size));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            weights_input_hidden[i][j] = dis(gen);

    for (int i = 0; i < hidden_size; ++i)
        for (int j = 0; j < output_size; ++j)
            weights_hidden_output[i][j] = dis(gen);
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    // Calculate hidden layer activations
    std::vector<double> hidden_layer(hidden_size, 0.0);
    for (int j = 0; j < hidden_size; ++j) {
        for (int i = 0; i < input_size; ++i) {
            hidden_layer[j] += input[i] * weights_input_hidden[i][j];
        }
        hidden_layer[j] = sigmoid(hidden_layer[j]);  // Apply activation function
    }

    // Calculate output layer activations
    std::vector<double> output_layer(output_size, 0.0);
    for (int k = 0; k < output_size; ++k) {
        for (int j = 0; j < hidden_size; ++j) {
            output_layer[k] += hidden_layer[j] * weights_hidden_output[j][k];
        }
        output_layer[k] = sigmoid(output_layer[k]);  // Apply activation function
    }

    return output_layer;
}
