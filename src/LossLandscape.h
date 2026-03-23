#pragma once

#include "NeuralNetwork.h"
#include "DataLoader.h"

#include <string>
#include <vector>

class LossLandscape {
public:
    static void compute(NeuralNetwork&             nn,
                        const std::vector<Sample>& samples,
                        const std::string&         outputPath,
                        int                        resolution = 40,
                        double                     range      = 1.0);

private:
    static void   filterNormalize(std::vector<double>& direction, const WeightSnapshot& snap);
    static void   applyPerturbation(NeuralNetwork& nn, const WeightSnapshot& center,
                                    const std::vector<double>& d1,
                                    const std::vector<double>& d2,
                                    double alpha, double beta);
    static double evalLoss(NeuralNetwork& nn, const std::vector<Sample>& samples);
};