#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    std::vector<double> forward(const std::vector<double>& input);

    // Add these getter methods
    int getInputSize() const { return input_size; }
    int getHiddenSize() const { return hidden_size; }
    int getOutputSize() const { return output_size; }

private:
    int input_size;
    int hidden_size;
    int output_size;
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;

    double sigmoid(double x);
    double sigmoid_derivative(double x);
};

#endif