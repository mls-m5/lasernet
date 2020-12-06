#pragma once

#include <vector>

//! Used to store working data for a whole network
class Data {
    std::vector<double> _data;

public:
    Data(size_t size) : _data(size, 0) {}
};
