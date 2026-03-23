#pragma once

#include <vector>
#include <string>
#include <utility>

struct Sample {
    std::vector<double> features;
    double              label;
};

struct Preprocessor {
    std::vector<double> mins;
    std::vector<double> ranges;

    double ageMean  = 0.0;
    double fareMean = 0.0;
};

class DataLoader {
public:
    // Loads and preprocesses train.csv. Fits the preprocessor on the full set.
    // Returns all samples in shuffled order, ready for k-fold splitting in the caller.
    static std::vector<Sample>
    loadAll(const std::string& csvPath, Preprocessor& outPreprocessor);

    // Loads the Kaggle test.csv (no Survived column).
    // Returns {features, passengerIds} in file order.
    static std::pair<std::vector<std::vector<double>>, std::vector<int>>
    loadTest(const std::string&  csvPath,
             const Preprocessor& preprocessor);

private:
    static double computeMean(const std::vector<std::vector<double>>& data, int col);

    static void applyImpute(std::vector<std::vector<double>>& data, int col, double mean);

    static void fitNormalize(const std::vector<std::vector<double>>& data,
                             int col, std::vector<double>& mins, std::vector<double>& ranges);

    static void applyNormalize(std::vector<std::vector<double>>& data,
                               int col, double mn, double range);
};