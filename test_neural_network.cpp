#include <gtest/gtest.h>
#include "neural_network.h"

TEST(NeuralNetworkTest, Initialization) {
    NeuralNetwork nn(3, 4, 2);
    EXPECT_EQ(nn.getInputSize(), 3);
    EXPECT_EQ(nn.getHiddenSize(), 4);
    EXPECT_EQ(nn.getOutputSize(), 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}