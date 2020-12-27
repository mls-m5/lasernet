#pragma once

#include "graph/inode.h"

#include "mls-unit-test/mock.h"

class MockNode : public INode {
public:
    MOCK_METHOD0(DataSize, dataSize, (), const override);
    MOCK_METHOD1(ConstSpanD, input, (ConstSpanD data), const override);
    MOCK_METHOD1(ConstSpanD, output, (ConstSpanD data), const override);
    MOCK_METHOD1(void, calculateValues, (CalculateArgs), const override);
    MOCK_METHOD1(void, backpropagate, (BackpropagateArgs), const override);
};
