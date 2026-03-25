#pragma once

#include "Neuron.h"

#include <vector>
#include <functional>

struct WeightSnapshot {
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes,
                  const std::vector<std::function<double(double)>>& activationFns,
                  const std::vector<std::function<double(double)>>& activationFnDerivatives);

    std::vector<double> forward(const std::vector<double>& inputs);
    void backward(const std::vector<double>& targets);
    void updateWeights(double learningRate);
    double loss(const std::vector<double>& targets) const;

    const std::vector<int>& getLayerSizes() const;

    WeightSnapshot saveWeights() const;
    void           loadWeights(const WeightSnapshot& snapshot);

private:
    std::vector<int> layerSizes;
    std::vector<std::vector<Neuron>> layers;

    // Cached from last forward pass
    std::vector<double> lastOutputs;
};