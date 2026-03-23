#include "LossLandscape.h"

#include <fstream>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

void LossLandscape::filterNormalize(std::vector<double>& direction,
                                    const WeightSnapshot& snap)
{
    int offset = 0;
    for (const auto& layer : snap.weights) {
        for (const auto& neuronWeights : layer) {
            int len = (int)neuronWeights.size();

            double trainedNorm = 0.0;
            for (double w : neuronWeights)
                trainedNorm += w * w;
            trainedNorm = std::sqrt(trainedNorm);

            double dirNorm = 0.0;
            for (int k = 0; k < len; k++)
                dirNorm += direction[offset + k] * direction[offset + k];
            dirNorm = std::sqrt(dirNorm);

            double scale = (dirNorm > 1e-10) ? trainedNorm / dirNorm : 0.0;
            for (int k = 0; k < len; k++)
                direction[offset + k] *= scale;

            offset += len;
        }
    }
}

void LossLandscape::applyPerturbation(NeuralNetwork& nn, const WeightSnapshot& center,
                                       const std::vector<double>& d1,
                                       const std::vector<double>& d2,
                                       double alpha, double beta)
{
    WeightSnapshot perturbed = center;

    int offset = 0;
    for (auto& layer : perturbed.weights) {
        for (auto& neuronWeights : layer) {
            for (double& w : neuronWeights) {
                w += alpha * d1[offset] + beta * d2[offset];
                offset++;
            }
        }
    }
    for (auto& layer : perturbed.biases) {
        for (double& b : layer) {
            b += alpha * d1[offset] + beta * d2[offset];
            offset++;
        }
    }

    nn.loadWeights(perturbed);
}

double LossLandscape::evalLoss(NeuralNetwork& nn, const std::vector<Sample>& samples)
{
    double total = 0.0;
    for (const Sample& s : samples) {
        nn.forward(s.features);
        total += nn.loss({ s.label });
    }
    return total / (double)samples.size();
}

void LossLandscape::compute(NeuralNetwork&             nn,
                             const std::vector<Sample>& samples,
                             const std::string&         outputPath,
                             int                        resolution,
                             double                     range)
{
    if (samples.empty())
        throw std::invalid_argument("samples must not be empty");

    const WeightSnapshot center = nn.saveWeights();

    int dim = 0;
    for (const auto& layer : center.weights)
        for (const auto& neuronWeights : layer)
            dim += (int)neuronWeights.size();
    for (const auto& layer : center.biases)
        dim += (int)layer.size();

    std::mt19937 rng(1234);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> d1(dim), d2(dim);
    for (double& v : d1) v = dist(rng);
    for (double& v : d2) v = dist(rng);

    filterNormalize(d1, center);
    filterNormalize(d2, center);

    std::ofstream out(outputPath);
    if (!out.is_open())
        throw std::runtime_error("Could not write loss landscape to: " + outputPath);

    out << "alpha,beta,loss\n";

    for (int row = 0; row < resolution; row++) {
        double beta = -range + 2.0 * range * row / (resolution - 1);

        for (int col = 0; col < resolution; col++) {
            double alpha = -range + 2.0 * range * col / (resolution - 1);

            applyPerturbation(nn, center, d1, d2, alpha, beta);
            double lossVal = evalLoss(nn, samples);

            out << alpha << "," << beta << "," << lossVal << "\n";
        }

        std::cout << "Loss landscape: row " << row + 1 << "/" << resolution << "\r" << std::flush;
    }
    std::cout << "\nLoss landscape written to: " << outputPath << "\n";

    nn.loadWeights(center);
}