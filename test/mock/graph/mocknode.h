#pragma once

#include "graph/inode.h"

#include "mls-unit-test/mock.h"

class MockNode : public INode {
public:
    MOCK_METHOD0(DataSize, dataSize, (), override);
    MOCK_METHOD1(ConstSpanD, input, (ConstSpanD data), override);
    MOCK_METHOD1(ConstSpanD, output, (ConstSpanD data), override);
    MOCK_METHOD1(void, calculateValues, (CalculateArgs), override);
    MOCK_METHOD1(void, backpropagate, (BackpropagateArgs), override);
};
