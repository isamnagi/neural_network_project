#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neural_network.h"

PYBIND11_MODULE(neural_network, m) {
    pybind11::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(pybind11::init<int, int, int>())
        .def("forward", &NeuralNetwork::forward);
}