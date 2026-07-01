#pragma once
#include "types.h"

// CNN 架构: Conv(1→8) → ReLU → Pool → Conv(8→16) → ReLU → Pool → FC(784→128) → FC(128→10)
class CNN {
public:
    CNN(double lr = 0.01, int batch_size = 32, int epochs = 10);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    vector<int> predict(const vector<FeatureVec>& X) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    // 将 784 维向量转成 28×28 的 2D 图（单通道）
    vector<vector<vector<double>>> to_image(const FeatureVec& x) const;

    // 卷积: input[C_in][H][W] → output[C_out][H][W] (padding=1, stride=1)
    void conv2d(const vector<vector<vector<double>>>& input,
                const vector<vector<vector<vector<double>>>>& filters,
                const vector<double>& bias,
                vector<vector<vector<double>>>& output) const;

    // 最大池化 2x2 stride=2
    void maxpool(const vector<vector<vector<double>>>& input,
                 vector<vector<vector<double>>>& output,
                 vector<vector<vector<vector<int>>>>& max_idx) const;

    // ReLU
    void relu(vector<vector<vector<double>>>& x) const;

    // 全连接层
    void fc_forward(const vector<double>& input,
                    const vector<vector<double>>& W,
                    const vector<double>& b,
                    vector<double>& output) const;

    void softmax(vector<double>& z) const;

    // 反向传播
    void conv2d_backward(const vector<vector<vector<double>>>& input,
                         const vector<vector<vector<double>>>& output_grad,
                         const vector<vector<vector<vector<double>>>>& filters,
                         vector<vector<vector<double>>>& input_grad,
                         vector<vector<vector<vector<double>>>>& filter_grad,
                         vector<double>& bias_grad) const;

    void maxpool_backward(const vector<vector<vector<double>>>& output_grad,
                          const vector<vector<vector<vector<int>>>>& max_idx,
                          vector<vector<vector<double>>>& input_grad) const;

    void relu_backward(const vector<vector<vector<double>>>& input,
                       vector<vector<vector<double>>>& grad) const;

    void fc_backward(const vector<double>& input,
                     const vector<double>& output_grad,
                     const vector<vector<double>>& W,
                     vector<double>& input_grad,
                     vector<vector<double>>& W_grad,
                     vector<double>& b_grad) const;

    // 参数
    double lr_;
    int batch_size_;
    int epochs_;

    // Conv1: 1→8, 3×3
    vector<vector<vector<vector<double>>>> W_conv1_; // [8][1][3][3]
    vector<double> b_conv1_;                          // [8]

    // Conv2: 8→16, 3×3
    vector<vector<vector<vector<double>>>> W_conv2_; // [16][8][3][3]
    vector<double> b_conv2_;                          // [16]

    // FC1: 7*7*16=784 → 128
    vector<vector<double>> W_fc1_;  // [784][128]
    vector<double> b_fc1_;          // [128]

    // FC2: 128 → 10
    vector<vector<double>> W_fc2_;  // [128][10]
    vector<double> b_fc2_;          // [10]
};
