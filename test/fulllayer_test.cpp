#include "layers/fulllayer.h"
#include "mls-unit-test/unittest.h"
#include "optimizer/trainer.h"

TEST_SUIT_BEGIN

TEST_CASE("create") {
    auto layer = FullLayer{2, 3};

    auto sizes = layer.dataSize();

    ASSERT_EQ(sizes.input, 2);
    ASSERT_EQ(sizes.parameters, 9);
    ASSERT_EQ(sizes.output, 3);
}

TEST_CASE("bias forward") {
    auto layer = FullLayer{2, 3};

    auto data = Trainer::BackPropagationData{layer.dataSize()};

    auto parameters = std::vector<double>(layer.dataSize().parameters);

    auto input = std::array{0., 0.};

    layer.bias(parameters, 0) = 1;
    layer.bias(parameters, 1) = 2;
    layer.bias(parameters, 2) = 3;

    layer.calculateValues({
        .x = input,
        .parameters = parameters,
        .y = data.y,
    });

    ASSERT_EQ(data.y.size(), 3);
    ASSERT_EQ(data.y[0], 1);
    ASSERT_EQ(data.y[1], 2);
    ASSERT_EQ(data.y[2], 3);
}

TEST_CASE("weights forward") {
    auto layer = FullLayer{2, 3};

    auto data = Trainer::BackPropagationData{layer.dataSize()};

    auto parameters = std::vector<double>(layer.dataSize().parameters);

    auto input = std::array{10., 20.};

    layer.weights(parameters, 0, 0) = 1;
    layer.weights(parameters, 0, 1) = 2;
    layer.weights(parameters, 0, 2) = 3;
    layer.weights(parameters, 1, 0) = 4;
    layer.weights(parameters, 1, 1) = 5;
    layer.weights(parameters, 1, 2) = 6;

    layer.calculateValues({
        .x = input,
        .parameters = parameters,
        .y = data.y,
    });

    ASSERT_EQ(data.y.size(), 3);
    ASSERT_EQ(data.y[0], 10. * 1. + 20. * 4.);
    ASSERT_EQ(data.y[1], 10. * 2. + 20. * 5.);
    ASSERT_EQ(data.y[2], 10. * 3. + 20. * 6.);
}

TEST_CASE("bias backpropagate") {
    auto layer = FullLayer{2, 3};

    auto data = Trainer::BackPropagationData{layer.dataSize()};

    auto parameters = std::vector<double>(layer.dataSize().parameters);

    auto input = std::array{0., 0.};

    data.dEdy[0] = 1;
    data.dEdy[1] = 2;
    data.dEdy[2] = 3;

    layer.backpropagate({
        .x = input,
        .parameters = parameters,
        .y = data.y,
        .dEdy = data.dEdy,
        .dEdx = data.dEdx,
        .dEdw = data.dEdw,
    });

    ASSERT_EQ(layer.cbias(data.dEdw, 0), 1);
    ASSERT_EQ(layer.cbias(data.dEdw, 1), 2);
    ASSERT_EQ(layer.cbias(data.dEdw, 2), 3);
}

TEST_CASE("weight backpropagate") {
    auto layer = FullLayer{2, 3};

    auto data = Trainer::BackPropagationData{layer.dataSize()};

    auto parameters = std::vector<double>(layer.dataSize().parameters);

    auto input = std::array{10., 20.};

    data.dEdy[0] = 1000;
    data.dEdy[1] = 2000;
    data.dEdy[2] = 3000;

    layer.weights(parameters, 0, 0) = 1;
    layer.weights(parameters, 0, 1) = 2;
    layer.weights(parameters, 0, 2) = 3;
    layer.weights(parameters, 1, 0) = 4;
    layer.weights(parameters, 1, 1) = 5;
    layer.weights(parameters, 1, 2) = 6;

    layer.backpropagate({
        .x = input,
        .parameters = parameters,
        .y = data.y,
        .dEdy = data.dEdy,
        .dEdx = data.dEdx,
        .dEdw = data.dEdw,
    });

    ASSERT_EQ(layer.cweights(data.dEdw, 0, 0), 1000 * 10);
    ASSERT_EQ(layer.cweights(data.dEdw, 0, 1), 2000 * 10);
    ASSERT_EQ(layer.cweights(data.dEdw, 0, 2), 3000 * 10);
    ASSERT_EQ(layer.cweights(data.dEdw, 1, 0), 1000 * 20);
    ASSERT_EQ(layer.cweights(data.dEdw, 1, 1), 2000 * 20);
    ASSERT_EQ(layer.cweights(data.dEdw, 1, 2), 3000 * 20);

    ASSERT_EQ(data.dEdx[0], 14000);
    ASSERT_EQ(data.dEdx[1], 32000);
}

TEST_SUIT_END
