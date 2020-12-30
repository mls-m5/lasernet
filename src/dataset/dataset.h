#pragma once

#include <vector>

struct Dataset {
    struct DataPair {
        std::vector<double> x;
        std::vector<double> y;
    };

    Dataset(std::vector<DataPair> data) : data(std::move(data)) {}

    Dataset() = default;
    Dataset(const Dataset &) = default;
    Dataset(Dataset &&) = default;
    Dataset &operator=(const Dataset &) = default;
    Dataset &operator=(Dataset &&) = default;

    std::vector<DataPair> data;
};

struct CombinedDataset {
    Dataset train;
    Dataset test;
};
