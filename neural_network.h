#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    void train(const std::vector<std::vector<double>>& training_data, const std::vector<std::vector<double>>& labels);
    void test(const std::vector<std::vector<double>>& test_data);

private:
    int input_size;
    int hidden_size;
    int output_size;
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;

    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& output, const std::vector<double>& target);
    double sigmoid(double x);
    double sigmoid_derivative(double x);
};

#endif
