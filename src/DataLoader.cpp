#include "DataLoader.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>

static std::vector<std::string> splitCSVLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

static double parseDouble(const std::string& s)
{
    if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
    try    { return std::stod(s); }
    catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

// Mr=0, Mrs=1, Miss=2, Master=3, Rare=4
static double extractTitle(const std::string& name)
{
    if (name.find("Mr.")     != std::string::npos) return 0.0;
    if (name.find("Mrs.")    != std::string::npos) return 1.0;
    if (name.find("Miss.")   != std::string::npos) return 2.0;
    if (name.find("Master.") != std::string::npos) return 3.0;
    return 4.0;
}

// Numeric-only=0, PC=1, CA=2, A5=3, STONO=4, Other=5
static double extractTicketPrefix(const std::string& ticket)
{
    size_t space = ticket.rfind(' ');
    if (space == std::string::npos) return 0.0;

    std::string prefix = ticket.substr(0, space);

    std::string clean;
    for (char c : prefix) {
        if (std::isalpha(c)) clean += std::toupper(c);
    }

    if (clean == "PC")    return 1.0;
    if (clean == "CA")    return 2.0;
    if (clean == "A5" || clean == "A")  return 3.0;
    if (clean == "STONO" || clean == "STONО2" || clean == "SOTON") return 4.0;
    return 5.0;
}

static std::vector<double> buildFeatureRow(const std::vector<std::string>& fields,
                                           int iPclass, int iSex,    int iAge,
                                           int iSibSp,  int iParch,
                                           int iFare,   int iEmbarked,
                                           int iName,   int iTicket)
{
    double sex  = (fields[iSex]      == "female") ? 1.0 : 0.0;
    double embS = (fields[iEmbarked] == "S")      ? 1.0 : 0.0;
    double embC = (fields[iEmbarked] == "C")      ? 1.0 : 0.0;

    double sibSp      = parseDouble(fields[iSibSp]);
    double parch      = parseDouble(fields[iParch]);
    double familySize = sibSp + parch + 1.0;
    double isAlone    = (familySize == 1.0) ? 1.0 : 0.0;

    double title        = extractTitle(fields[iName]);
    double ticketPrefix = extractTicketPrefix(fields[iTicket]);

    return {
        parseDouble(fields[iPclass]),
        sex,
        parseDouble(fields[iAge]),
        sibSp,
        parch,
        parseDouble(fields[iFare]),
        embS,
        embC,
        title,
        familySize,
        isAlone,
        ticketPrefix
    };
}

double DataLoader::computeMean(const std::vector<std::vector<double>>& data, int col)
{
    double sum   = 0.0;
    int    count = 0;

    for (const auto& row : data) {
        if (!std::isnan(row[col])) {
            sum += row[col];
            ++count;
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

void DataLoader::applyImpute(std::vector<std::vector<double>>& data, int col, double mean)
{
    for (auto& row : data)
        if (std::isnan(row[col]))
            row[col] = mean;
}

void DataLoader::fitNormalize(const std::vector<std::vector<double>>& data,
                              int col, std::vector<double>& mins, std::vector<double>& ranges)
{
    double mn =  std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();

    for (const auto& row : data) {
        mn = std::min(mn, row[col]);
        mx = std::max(mx, row[col]);
    }

    mins[col]   = mn;
    ranges[col] = mx - mn;
}

void DataLoader::applyNormalize(std::vector<std::vector<double>>& data,
                                int col, double mn, double range)
{
    if (range == 0.0) return;
    for (auto& row : data)
        row[col] = (row[col] - mn) / range;
}

std::vector<Sample>
DataLoader::loadAll(const std::string& csvPath, Preprocessor& outPreprocessor)
{
    std::ifstream file(csvPath);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + csvPath);

    std::string headerLine;
    std::getline(file, headerLine);
    std::vector<std::string> headers = splitCSVLine(headerLine);

    auto colIndex = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)headers.size(); i++)
            if (headers[i] == name) return i;
        throw std::runtime_error("Column not found: " + name);
    };

    const int iSurvived = colIndex("Survived");
    const int iPclass   = colIndex("Pclass");
    const int iName     = colIndex("Name");
    const int iSex      = colIndex("Sex");
    const int iAge      = colIndex("Age");
    const int iSibSp    = colIndex("SibSp");
    const int iParch    = colIndex("Parch");
    const int iTicket   = colIndex("Ticket");
    const int iFare     = colIndex("Fare");
    const int iEmbarked = colIndex("Embarked");

    std::vector<std::vector<double>> X;
    std::vector<double>              y;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> fields = splitCSVLine(line);
        if ((int)fields.size() <= std::max({iSurvived, iPclass, iName, iSex, iAge, iSibSp, iParch, iTicket, iFare, iEmbarked}))
            continue;

        double survived = parseDouble(fields[iSurvived]);
        if (std::isnan(survived)) continue;

        X.push_back(buildFeatureRow(fields, iPclass, iSex, iAge, iSibSp, iParch, iFare, iEmbarked, iName, iTicket));
        y.push_back(survived);
    }

    if (X.empty())
        throw std::runtime_error("No valid rows found in: " + csvPath);

    outPreprocessor.ageMean  = computeMean(X, 2);
    outPreprocessor.fareMean = computeMean(X, 5);

    applyImpute(X, 2, outPreprocessor.ageMean);
    applyImpute(X, 5, outPreprocessor.fareMean);

    outPreprocessor.mins.resize(12, 0.0);
    outPreprocessor.ranges.resize(12, 1.0);

    for (int col : { 0, 2, 3, 4, 5, 8, 9, 11 })
        fitNormalize(X, col, outPreprocessor.mins, outPreprocessor.ranges);

    for (int col : { 0, 2, 3, 4, 5, 8, 9, 11 })
        applyNormalize(X, col, outPreprocessor.mins[col], outPreprocessor.ranges[col]);

    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<Sample> samples;
    samples.reserve(X.size());
    for (int idx : indices)
        samples.push_back({ X[idx], y[idx] });

    return samples;
}

std::pair<std::vector<std::vector<double>>, std::vector<int>>
DataLoader::loadTest(const std::string&  csvPath,
                     const Preprocessor& preprocessor)
{
    std::ifstream file(csvPath);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + csvPath);

    std::string headerLine;
    std::getline(file, headerLine);
    std::vector<std::string> headers = splitCSVLine(headerLine);

    auto colIndex = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)headers.size(); i++)
            if (headers[i] == name) return i;
        throw std::runtime_error("Column not found: " + name);
    };

    const int iPassengerId = colIndex("PassengerId");
    const int iPclass      = colIndex("Pclass");
    const int iName        = colIndex("Name");
    const int iSex         = colIndex("Sex");
    const int iAge         = colIndex("Age");
    const int iSibSp       = colIndex("SibSp");
    const int iParch       = colIndex("Parch");
    const int iTicket      = colIndex("Ticket");
    const int iFare        = colIndex("Fare");
    const int iEmbarked    = colIndex("Embarked");

    std::vector<std::vector<double>> X;
    std::vector<int>                 passengerIds;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> fields = splitCSVLine(line);
        if ((int)fields.size() <= std::max({iPassengerId, iPclass, iName, iSex, iAge, iSibSp, iParch, iTicket, iFare, iEmbarked}))
            continue;

        passengerIds.push_back((int)parseDouble(fields[iPassengerId]));
        X.push_back(buildFeatureRow(fields, iPclass, iSex, iAge, iSibSp, iParch, iFare, iEmbarked, iName, iTicket));
    }

    if (X.empty())
        throw std::runtime_error("No valid rows found in: " + csvPath);

    applyImpute(X, 2, preprocessor.ageMean);
    applyImpute(X, 5, preprocessor.fareMean);

    for (int col : { 0, 2, 3, 4, 5, 8, 9, 11 })
        applyNormalize(X, col, preprocessor.mins[col], preprocessor.ranges[col]);

    return { std::move(X), std::move(passengerIds) };
}