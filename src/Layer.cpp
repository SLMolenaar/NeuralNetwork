#include "Layer.h"

#include <random>
#include <cmath>
#include <stdexcept>

Layer::Layer(int numInputs, int numOutputs,
             std::function<double(double)> activationFn,
             std::function<double(double)> activationFnDerivative)
    : nIn(numInputs), nOut(numOutputs),
      weights(numOutputs, numInputs),
      biases(Eigen::VectorXd::Zero(numOutputs)),
      gradWeights(Eigen::MatrixXd::Zero(numOutputs, numInputs)),
      gradBiases(Eigen::VectorXd::Zero(numOutputs)),
      cachedInputs(numInputs),
      cachedOutputs(numOutputs),
      activationFn(std::move(activationFn)),
      activationFnDerivative(std::move(activationFnDerivative))
{
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / numInputs));

    for (int i = 0; i < nOut; ++i)
        for (int j = 0; j < nIn; ++j)
            weights(i, j) = dist(rng);
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd& inputs)
{
    if (inputs.size() != nIn)
        throw std::invalid_argument("Input size does not match layer's expected input count");

    cachedInputs = inputs;

    cachedOutputs = (weights * inputs + biases).unaryExpr(activationFn);

    return cachedOutputs;
}

Eigen::VectorXd Layer::backward(const Eigen::VectorXd& upstreamGrad)
{
    if (upstreamGrad.size() != nOut)
        throw std::invalid_argument("Upstream gradient size does not match layer output count");

    const Eigen::VectorXd delta = upstreamGrad.cwiseProduct(
        cachedOutputs.unaryExpr(activationFnDerivative));

    gradWeights += delta * cachedInputs.transpose();
    gradBiases  += delta;

    return weights.transpose() * delta;
}

void Layer::updateWeights(double learningRate)
{
    weights     -= learningRate * gradWeights;
    biases      -= learningRate * gradBiases;
    gradWeights.setZero();
    gradBiases.setZero();
}

void Layer::setWeightsAndBiases(const Eigen::MatrixXd& w, const Eigen::VectorXd& b)
{
    weights = w;
    biases  = b;
}
