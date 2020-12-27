#include "layers/singlestraightline.h"
#include "mls-unit-test/unittest.h"
#include <array>

constexpr double eps = std::numeric_limits<double>::min();

TEST_SUIT_BEGIN

TEST_CASE("create") {
    auto line = SingleStraightLine{};

    auto sizes = line.dataSize();
    ASSERT_EQ(sizes.parameters, 2);
    ASSERT_EQ(sizes.input, 1);
    ASSERT_EQ(sizes.output, 1);
}

TEST_CASE("forward") {
    auto line = SingleStraightLine{};

    auto y = std::array{0.};

    line.calculateValues({
        .x = {{1.}},
        .parameters = {{2., 3.}},
        .y = y,
    });

    ASSERT_NEAR(y.front(), 2. * 1. + 3., eps);
}

TEST_CASE("backpropagate") {
    auto line = SingleStraightLine{};

    auto dEdw = std::array{0., 0.};
    auto dEdx = std::array{0.};

    line.backpropagate({
        .x = {{1.}},
        .parameters = {{2., 3.}},
        .y = {{5.}},
        .dEdxPrev = {{10.}},
        .dEdx = dEdx,
        .dEdw = dEdw,
    });

    ASSERT_NEAR(dEdw.front(), 20., eps);
    ASSERT_NEAR(dEdw.back(), 10., eps);

    ASSERT_NEAR(dEdx.front(), 20., eps);
}

TEST_SUIT_END
