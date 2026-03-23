#pragma once

#include <vector>
#include <functional>

class Neuron {
public:
    Neuron(int numInputs,
           std::function<double(double)> activationFn,
           std::function<double(double)> activationFnDerivative);

    double forward(const std::vector<double>& inputs);
    void   backward(double delta);
    void   updateWeights(double learningRate);

    double getWeight(int index) const { return weights[index]; }
    double getBias()            const { return bias; }

    void setWeightsAndBias(const std::vector<double>& w, double b) { weights = w; bias = b; }

    double delta;

private:
    std::vector<double> weights;
    std::vector<double> gradWeights;
    double bias;
    double gradBias;

    // Cached from forward pass, needed during backward
    std::vector<double> inputs;
    double output;

    std::function<double(double)> activationFn;
    std::function<double(double)> activationFnDerivative;
};
