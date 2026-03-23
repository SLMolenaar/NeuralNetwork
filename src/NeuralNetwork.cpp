#include "NeuralNetwork.h"

#include <stdexcept>
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::vector<std::function<double(double)>>& activationFns,
                             const std::vector<std::function<double(double)>>& activationFnDerivatives)
    : layerSizes(layerSizes)
{
    if (layerSizes.size() < 2)
        throw std::invalid_argument("Network must have at least 2 layers (input and output)");

    if (activationFns.size() != layerSizes.size() - 1)
        throw std::invalid_argument("activationFns must have one entry per layer excluding input");

    if (activationFnDerivatives.size() != activationFns.size())
        throw std::invalid_argument("activationFnDerivatives must match activationFns in size");

    // layerSizes[0] is the input size, not an actual neuron layer
    for (int i = 1; i < (int)layerSizes.size(); i++) {
        std::vector<Neuron> layer;
        layer.reserve(layerSizes[i]);

        for (int j = 0; j < layerSizes[i]; j++)
            layer.emplace_back(layerSizes[i - 1], activationFns[i - 1], activationFnDerivatives[i - 1]);

        layers.push_back(std::move(layer));
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& inputs)
{
    std::vector<double> activations = inputs;

    for (std::vector<Neuron>& layer : layers) {
        std::vector<double> nextActivations;
        nextActivations.reserve(layer.size());

        for (Neuron& neuron : layer)
            nextActivations.push_back(neuron.forward(activations));

        activations = std::move(nextActivations);
    }

    lastOutputs = activations;
    return activations;
}

void NeuralNetwork::backward(const std::vector<double>& targets)
{
    if (targets.size() != lastOutputs.size())
        throw std::invalid_argument("Targets size does not match output size");

    std::vector<Neuron>& outputLayer = layers.back();
    for (int i = 0; i < (int)outputLayer.size(); i++)
        outputLayer[i].backward(lastOutputs[i] - targets[i]);

    for (int i = (int)layers.size() - 2; i >= 0; i--) {
        std::vector<Neuron>& currentLayer = layers[i];
        std::vector<Neuron>& nextLayer    = layers[i + 1];

        for (int j = 0; j < (int)currentLayer.size(); j++) {
            double error = 0.0;
            for (Neuron& nextNeuron : nextLayer)
                error += nextNeuron.delta * nextNeuron.getWeight(j);

            currentLayer[j].backward(error);
        }
    }
}

void NeuralNetwork::updateWeights(double learningRate)
{
    for (std::vector<Neuron>& layer : layers)
        for (Neuron& neuron : layer)
            neuron.updateWeights(learningRate);
}

double NeuralNetwork::loss(const std::vector<double>& targets) const
{
    if (targets.size() != lastOutputs.size())
        throw std::invalid_argument("Targets size does not match output size");

    double sum = 0.0;
    for (int i = 0; i < (int)targets.size(); i++) {
        double diff = lastOutputs[i] - targets[i];
        sum += diff * diff;
    }

    return sum / targets.size();
}

const std::vector<int>& NeuralNetwork::getLayerSizes() const
{
    return layerSizes;
}

WeightSnapshot NeuralNetwork::saveWeights() const
{
    WeightSnapshot snapshot;
    snapshot.weights.resize(layers.size());
    snapshot.biases.resize(layers.size());

    for (int i = 0; i < (int)layers.size(); i++) {
        snapshot.weights[i].resize(layers[i].size());
        snapshot.biases[i].resize(layers[i].size());

        for (int j = 0; j < (int)layers[i].size(); j++) {
            const Neuron& neuron = layers[i][j];
            snapshot.biases[i][j] = neuron.getBias();

            snapshot.weights[i][j].resize(layerSizes[i]);
            for (int k = 0; k < layerSizes[i]; k++)
                snapshot.weights[i][j][k] = neuron.getWeight(k);
        }
    }

    return snapshot;
}

void NeuralNetwork::loadWeights(const WeightSnapshot& snapshot)
{
    for (int i = 0; i < (int)layers.size(); i++)
        for (int j = 0; j < (int)layers[i].size(); j++)
            layers[i][j].setWeightsAndBias(snapshot.weights[i][j], snapshot.biases[i][j]);
}
