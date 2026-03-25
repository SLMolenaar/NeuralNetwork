#include "NeuralNetwork.h"

#include <stdexcept>
#include <cmath>

NeuralNetwork::NeuralNetwork(
    const std::vector<int>& layerSizes,
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

    layers.reserve(layerSizes.size() - 1);
    for (int i = 1; i < (int)layerSizes.size(); ++i)
        layers.emplace_back(layerSizes[i - 1], layerSizes[i],
                            activationFns[i - 1], activationFnDerivatives[i - 1]);
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& inputs)
{
    Eigen::VectorXd activations = Eigen::Map<const Eigen::VectorXd>(inputs.data(), inputs.size());

    for (Layer& layer : layers)
        activations = layer.forward(activations);

    lastOutputs = std::vector<double>(activations.data(), activations.data() + activations.size());
    return lastOutputs;
}

void NeuralNetwork::backward(const std::vector<double>& targets)
{
    if (targets.size() != lastOutputs.size())
        throw std::invalid_argument("Targets size does not match output size");

    Eigen::VectorXd grad(lastOutputs.size());
    for (int i = 0; i < (int)lastOutputs.size(); ++i)
        grad[i] = lastOutputs[i] - targets[i];

    for (int i = (int)layers.size() - 1; i >= 0; --i)
        grad = layers[i].backward(grad);
}

void NeuralNetwork::updateWeights(double learningRate)
{
    for (Layer& layer : layers)
        layer.updateWeights(learningRate);
}

double NeuralNetwork::loss(const std::vector<double>& targets) const
{
    if (targets.size() != lastOutputs.size())
        throw std::invalid_argument("Targets size does not match output size");

    double sum = 0.0;
    for (int i = 0; i < (int)targets.size(); ++i) {
        double diff = lastOutputs[i] - targets[i];
        sum += diff * diff;
    }
    return sum / (double)targets.size();
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

    for (int i = 0; i < (int)layers.size(); ++i) {
        const Eigen::MatrixXd& W = layers[i].getWeights();
        const Eigen::VectorXd& b = layers[i].getBiases();

        const int nOut = (int)W.rows();
        const int nIn  = (int)W.cols();

        snapshot.weights[i].resize(nOut, std::vector<double>(nIn));
        snapshot.biases[i].resize(nOut);

        for (int r = 0; r < nOut; ++r) {
            snapshot.biases[i][r] = b[r];
            for (int c = 0; c < nIn; ++c)
                snapshot.weights[i][r][c] = W(r, c);
        }
    }
    return snapshot;
}

void NeuralNetwork::loadWeights(const WeightSnapshot& snapshot)
{
    for (int i = 0; i < (int)layers.size(); ++i) {
        const int nOut = (int)snapshot.weights[i].size();
        const int nIn  = (int)snapshot.weights[i][0].size();

        Eigen::MatrixXd W(nOut, nIn);
        Eigen::VectorXd b(nOut);

        for (int r = 0; r < nOut; ++r) {
            b[r] = snapshot.biases[i][r];
            for (int c = 0; c < nIn; ++c)
                W(r, c) = snapshot.weights[i][r][c];
        }

        layers[i].setWeightsAndBiases(W, b);
    }
}

