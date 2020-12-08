#pragma once

#include "network.h"

class InputLayer : public IInput {
    ConstSpanD _activation;

public:
    // This is expected to be set on every change of input sample
    void setInputData(SpanD activation) {
        _activation = activation;
    }

    //! @see IInput
    virtual ConstSpanD input() override {
        return _activation;
    }
};
