#include "src/NeuralNetwork.h"
#include "src/Visualizer.h"
#include "src/DataLoader.h"
#include "src/LossLandscape.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

static double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

static double sigmoidDerivative(double x)
{
    return x * (1.0 - x);
}

static double relu(double x)
{
    return std::max(0.0, x);
}

static double reluDerivative(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}

static NeuralNetwork makeNetwork()
{
    return NeuralNetwork(
        { 12, 16, 8, 1 },
        { relu, relu, sigmoid },
        { reluDerivative, reluDerivative, sigmoidDerivative }
    );
}

static double accuracy(NeuralNetwork& nn, const std::vector<Sample>& samples)
{
    int correct = 0;
    for (const Sample& s : samples) {
        std::vector<double> out = nn.forward(s.features);
        if ((out[0] >= 0.5 ? 1.0 : 0.0) == s.label) ++correct;
    }
    return (double)correct / (double)samples.size();
}

static double computeValLoss(NeuralNetwork& nn, const std::vector<Sample>& samples)
{
    double total = 0.0;
    for (const Sample& s : samples) {
        nn.forward(s.features);
        total += nn.loss({ s.label });
    }
    return total / samples.size();
}

static NeuralNetwork trainFold(const std::vector<Sample>& trainSet,
                               const std::vector<Sample>& valSet)
{
    constexpr int    maxEpochs    = 1000;
    constexpr double learningRate = 0.005;
    constexpr int    batchSize    = 32;
    constexpr int    patience     = 50;

    NeuralNetwork nn = makeNetwork();

    int            epochsNoImprovement = 0;
    double         bestValLoss         = std::numeric_limits<double>::infinity();
    WeightSnapshot bestWeights         = nn.saveWeights();

    std::mt19937 rng(42);

    std::vector<int> indices(trainSet.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 1; epoch <= maxEpochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int batchStart = 0; batchStart < (int)trainSet.size(); batchStart += batchSize) {
            int batchEnd = std::min(batchStart + batchSize, (int)trainSet.size());

            for (int i = batchStart; i < batchEnd; i++) {
                const Sample& s = trainSet[indices[i]];
                nn.forward(s.features);
                nn.backward({ s.label });
            }

            nn.updateWeights(learningRate);
        }

        double currentValLoss = computeValLoss(nn, valSet);

        if (currentValLoss < bestValLoss) {
            bestValLoss         = currentValLoss;
            bestWeights         = nn.saveWeights();
            epochsNoImprovement = 0;
        } else {
            ++epochsNoImprovement;
        }

        if (epochsNoImprovement >= patience) {
            nn.loadWeights(bestWeights);
            break;
        }
    }

    return nn;
}

static void writeSubmission(const std::vector<std::vector<double>>& testFeatures,
                            const std::vector<int>&                 passengerIds,
                            std::vector<NeuralNetwork>&             foldModels,
                            const std::string&                      outputPath)
{
    std::ofstream out(outputPath);
    if (!out.is_open())
        throw std::runtime_error("Could not write submission to: " + outputPath);

    out << "PassengerId,Survived\n";

    for (int i = 0; i < (int)passengerIds.size(); i++) {
        double prob = 0.0;
        for (NeuralNetwork& nn : foldModels)
            prob += nn.forward(testFeatures[i])[0];
        prob /= foldModels.size();

        out << passengerIds[i] << "," << (prob >= 0.5 ? 1 : 0) << "\n";
    }

    std::cout << "Submission written to: " << outputPath << "\n";
}

int main()
{
    constexpr int k = 5;

    Preprocessor preprocessor;
    std::vector<Sample> allSamples = DataLoader::loadAll("data/train.csv", preprocessor);
    std::cout << "Total samples: " << allSamples.size() << "  Folds: " << k << "\n\n";

    int foldSize = (int)allSamples.size() / k;

    std::vector<NeuralNetwork> foldModels;
    foldModels.reserve(k);

    double totalValAcc = 0.0;

    for (int fold = 0; fold < k; fold++) {
        int valStart = fold * foldSize;
        int valEnd   = (fold == k - 1) ? (int)allSamples.size() : valStart + foldSize;

        std::vector<Sample> valSet(allSamples.begin() + valStart, allSamples.begin() + valEnd);
        std::vector<Sample> trainSet;
        trainSet.reserve(allSamples.size() - valSet.size());
        trainSet.insert(trainSet.end(), allSamples.begin(), allSamples.begin() + valStart);
        trainSet.insert(trainSet.end(), allSamples.begin() + valEnd, allSamples.end());

        std::cout << "Fold " << fold + 1 << "/" << k
                  << "  train: " << trainSet.size()
                  << "  val: "   << valSet.size() << "\n";

        NeuralNetwork nn = trainFold(trainSet, valSet);

        double valAcc = accuracy(nn, valSet);
        totalValAcc += valAcc;
        std::cout << "Fold " << fold + 1 << " val accuracy: " << valAcc << "\n\n";

        foldModels.push_back(std::move(nn));
    }

    std::cout << "Mean val accuracy: " << totalValAcc / k << "\n\n";

    std::cout << "loss landscape\n";
    LossLandscape::compute(foldModels[0], allSamples, "../data/loss_landscape.csv", 40, 1.0);

    auto [testFeatures, passengerIds] = DataLoader::loadTest("data/test.csv", preprocessor);
    std::cout << "Kaggle test samples: " << testFeatures.size() << "\n";

    writeSubmission(testFeatures, passengerIds, foldModels, "../data/submission.csv");

    Visualizer vis(foldModels[0]);
    vis.show();

    return 0;
}