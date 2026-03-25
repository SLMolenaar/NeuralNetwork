#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>

class Layer {
public:
    Layer(int numInputs, int numOutputs,
          std::function<double(double)> activationFn,
          std::function<double(double)> activationFnDerivative);

    Eigen::VectorXd forward(const Eigen::VectorXd& inputs);

    Eigen::VectorXd backward(const Eigen::VectorXd& upstreamGrad);

    void updateWeights(double learningRate);

    int getNumInputs()  const { return nIn; }
    int getNumOutputs() const { return nOut; }

    const Eigen::MatrixXd& getWeights() const { return weights; }
    const Eigen::VectorXd& getBiases()  const { return biases; }
    void setWeightsAndBiases(const Eigen::MatrixXd& w, const Eigen::VectorXd& b);

private:
    int nIn, nOut;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::MatrixXd gradWeights;
    Eigen::VectorXd gradBiases;

    Eigen::VectorXd cachedInputs;
    Eigen::VectorXd cachedOutputs;

    std::function<double(double)> activationFn;
    std::function<double(double)> activationFnDerivative;
};
