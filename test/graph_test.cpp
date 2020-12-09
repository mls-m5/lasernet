#include "graph/graph.h"
#include "layers/singlestraightline.h"
#include "mls-unit-test/unittest.h"
#include "mock/graph/mocknode.h"
#include <array>

constexpr double eps = std::numeric_limits<double>::min();

TEST_SUIT_BEGIN

TEST_CASE("create") {
    auto graph = Graph{};
}

TEST_CASE("properties") {
    auto linea = SingleStraightLine{};
    auto lineb = SingleStraightLine{};

    auto mockline = MockNode{};

    auto nodes = std::vector<INode *>{&linea, &lineb};

    auto graph = Graph{nodes};

    ASSERT_EQ(graph.parameterSize(),
              linea.parameterSize() + lineb.parameterSize());

    ASSERT_EQ(graph.activationSize(),
              linea.activationSize() + lineb.activationSize());

    auto activationData = std::vector<double>{1., 2.};
    auto output = graph.output(activationData);

    ASSERT_EQ(output.size(), 1);
    ASSERT_NEAR(output.front(), 2., eps);
}

TEST_SUIT_END
