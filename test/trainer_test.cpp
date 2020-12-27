#include "dataset/dataset.h"
#include "fmt/core.h"
#include "layers/linearcost.h"
#include "layers/singlebias.h"
#include "mls-unit-test/unittest.h"
#include "msl/range.h"
#include "optimizer/gradientdescent.h"
#include "optimizer/trainer.h"

TEST_SUIT_BEGIN

TEST_CASE("create") {
    auto dataset = Dataset{{{{1}, {2}}}};

    auto optimizer = GradientDescent{};

    auto bias = SingleBias{};

    auto trainer = Trainer{bias, optimizer, dataset};
}

TEST_CASE("step") {
    auto dataset = Dataset{{{{1}, {2}}}};

    auto bias = SingleBias{};

    auto optimizer = GradientDescent{};

    auto trainer = Trainer{bias, optimizer, dataset};

    auto cost = LinearCost{};

    trainer.step(bias, cost);
}

TEST_CASE("expect loss to decrease") {
    auto dataset = Dataset{{{{1}, {2}}}};

    auto bias = SingleBias{};

    auto optimizer = GradientDescent{};

    auto trainer = Trainer{bias, optimizer, dataset};

    auto cost = LinearCost{};

    trainer.step(bias, cost);

    auto startCost = trainer.cost();

    fmt::print("start cost: {}\n", startCost);

    double lastCost = 100000000000000000000000000000.;

    for (auto n : msl::range(20)) {
        (void)n;

        trainer.step(bias, cost);

        lastCost = trainer.cost();

        fmt::print("cost: {}\n", lastCost);
    }

    ASSERT_LT(lastCost, startCost);
}

TEST_SUIT_END
