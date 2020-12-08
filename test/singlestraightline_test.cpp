#include "layers/singlestraightline.h"
#include "mls-unit-test/unittest.h"
#include <array>

constexpr double eps = std::numeric_limits<double>::min();

TEST_SUIT_BEGIN

TEST_CASE("create") {
    auto line = SingleStraightLine{};

    ASSERT_EQ(line.parameterSize(), 2);
    ASSERT_EQ(line.activationSize(), 1);
}

TEST_CASE("forward") {
    auto line = SingleStraightLine{};

    auto y = std::array{0.};

    line.calculateValues({{1.}}, {{2., 3.}}, y);

    ASSERT_NEAR(y.front(), 2. * 1. + 3., eps);
}

TEST_CASE("backpropagate") {
    auto line = SingleStraightLine{};

    auto derivative = std::array{0.};

    line.backpropagate({{1.}}, {{2., 3.}}, {{5.}}, {{10.}}, derivative);

    ASSERT_NEAR(derivative.front(), 2. * 10., eps);
}

TEST_SUIT_END
