#pragma once
#include "types.h"

// 全连接神经网络：输入 784 → 隐藏 128 → 输出 10
class NeuralNetwork {
public:
    NeuralNetwork(int input_size = 784, int hidden_size = 128, int output_size = 10,
                  double lr = 0.1, int batch_size = 64, int epochs = 20);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    vector<int> predict(const vector<FeatureVec>& X) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    // 激活函数及其导数
    double sigmoid(double z) const;
    double sigmoid_derivative(double z) const;  // 假设 z = sigmoid 的输出
    void softmax(vector<double>& z) const;

    // 核心训练步骤
    void forward(const FeatureVec& x,
                 vector<double>& h,        // 隐藏层输出
                 vector<double>& out) const; // 输出层概率

    void backward(const FeatureVec& x,
                  const vector<double>& y_true,
                  const vector<double>& h,
                  const vector<double>& out,
                  vector<vector<double>>& dW1,
                  vector<double>& db1,
                  vector<vector<double>>& dW2,
                  vector<double>& db2);

    int input_size_;
    int hidden_size_;
    int output_size_;
    double lr_;
    int batch_size_;
    int epochs_;

    // 网络参数
    vector<vector<double>> W1_;   // [input_size × hidden_size]
    vector<double> b1_;           // [hidden_size]
    vector<vector<double>> W2_;   // [hidden_size × output_size]
    vector<double> b2_;           // [output_size]
};
