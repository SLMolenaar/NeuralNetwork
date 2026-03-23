#include "Neuron.h"

#include <random>
#include <numeric>
#include <stdexcept>

Neuron::Neuron(int numInputs,
               std::function<double(double)> activationFn,
               std::function<double(double)> activationFnDerivative)
    : delta(0.0), bias(0.0), gradBias(0.0), output(0.0),
      activationFn(activationFn),
      activationFnDerivative(activationFnDerivative)
{
    // He initialization
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / numInputs));

    weights.resize(numInputs);
    gradWeights.resize(numInputs, 0.0);
    for (double& w : weights)
        w = dist(rng);
}

double Neuron::forward(const std::vector<double>& inputs)
{
    if (inputs.size() != weights.size())
        throw std::invalid_argument("Input size does not match number of weights");

    this->inputs = inputs;

    double sum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias);
    output = activationFn(sum);

    return output;
}

void Neuron::backward(double delta)
{
    this->delta = delta * activationFnDerivative(output);

    for (int i = 0; i < (int)weights.size(); i++)
        gradWeights[i] += this->delta * inputs[i];

    gradBias += this->delta;
}

void Neuron::updateWeights(double learningRate)
{
    for (int i = 0; i < (int)weights.size(); i++) {
        weights[i] -= learningRate * gradWeights[i];
        gradWeights[i] = 0.0;
    }

    bias     -= learningRate * gradBias;
    gradBias  = 0.0;
}
