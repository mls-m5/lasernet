#pragma once

#include <vector>

class Dataset {
public:
    struct DataPair {
        std::vector<double> x;
        std::vector<double> y;
    };

    Dataset(std::vector<DataPair> data) : data(std::move(data)) {}

    std::vector<DataPair> data;
};

class CombinedDataset {
    Dataset train;
    Dataset test;
};
