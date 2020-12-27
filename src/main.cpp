#include "fmt/core.h"
#include "layers/linearcost.h"
#include "layers/singlestraightline.h"
#include "optimizer/gradientdescent.h"
#include "optimizer/trainer.h"
#include <iostream>

const auto datapair = Dataset::DataPair{{1}, {2}};

const auto d = Dataset{{datapair}};

// x and y samples from a straight line
const auto dataset = Dataset{{
    {{6.6177992}, {1.9853397}},     {{4.23292517}, {1.26987755}},
    {{9.619341176}, {2.88580235}},  {{8.855802612}, {2.65674078}},
    {{1.700738511}, {0.510221553}}, {{0.3829222685}, {0.114876680}},
    {{7.114068058}, {2.13422041}},  {{9.000982418}, {2.70029472}},
    {{4.802254506}, {1.44067635}},  {{7.645870368}, {2.29376111}},
    {{9.309425381}, {2.79282761}},  {{4.640643418}, {1.39219302}},
    {{8.011992578}, {2.40359777}},  {{0.7601078724}, {0.228032361}},
    {{2.421176793}, {0.726353037}}, {{1.374340668}, {0.412302200}},
    {{7.054336104}, {2.11630083}},  {{9.147601815}, {2.74428054}},
    {{5.610451446}, {1.68313543}},  {{6.995655603}, {2.09869668}},
    {{7.994506}, {2.398351}},       {{3.041760471}, {0.912528141}},
    {{2.555377993}, {0.76661339}},  {{3.403311113}, {1.02099333}},
    {{4.195108331}, {1.25853249}},  {{0.1763334206}, {0.0529000261}},
    {{8.889854943}, {2.66695648}},  {{8.646111232}, {2.59383336}},
    {{4.848688841}, {1.45460665}},  {{4.511645321}, {1.353493596}},
}};

int main(int, char **) {
    // Test workflow:

    // LoadInput
    // auto data = load("mnist-for-example")

    //    auto data = Dataset{{1, 2}}; // Todo: Fill with actual data

    auto optimizer = GradientDescent{};

    auto node = SingleStraightLine{}; // Or network in the future

    auto trainer = Trainer{node, optimizer, dataset, dataset.data.size()};

    auto cost = LinearCost{};

    for (size_t i = 0; i < 2000; ++i) {
        trainer.step(node, cost);

        fmt::print("step: {}\t cost: {}\n", i, trainer.cost());
        fmt::print("\u001b[34;1m   k = {}, m = {} \u001b[0m\n",
                   trainer.parameters().front(),
                   trainer.parameters().back());
    }

    return 0;
}
