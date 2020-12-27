#include "layers/relu.h"
#include "mls-unit-test/unittest.h"

TEST_SUIT_BEGIN

TEST_CASE("create") {
    auto relu = Relu{10};
    auto sizes = relu.dataSize();

    ASSERT_EQ(sizes.input, 10);
    ASSERT_EQ(sizes.parameters, 0);
    ASSERT_EQ(sizes.output, 10);
}

TEST_CASE("forward when more than 0") {
    auto relu = Relu{2};
    auto res = std::array{0., 0.};

    relu.calculateValues({
        .x = {{-3., 4.}},
        .parameters = {},
        .y = res,
    });

    ASSERT_EQ(res.front(), 0);
    ASSERT_EQ(res.back(), 4.);
}

TEST_SUIT_END
