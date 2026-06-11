#pragma once
#include "types.h"

// 单个二分类器：判断"是不是某个数字"
class LogisticBinary {
public:
    LogisticBinary(double lr = 0.1, int epochs = 20);

    // 训练：X 是图片，y 是 0/1 标签
    void fit(const vector<FeatureVec>& X, const vector<int>& y);

    // 预测概率：返回 0~1
    vector<double> predict_proba(const vector<FeatureVec>& X) const;

private:
    double sigmoid(double z) const;

    vector<double> w_;   // 784 个权重
    double b_;           // 偏置
    double lr_;          // 学习率
    int epochs_;         // 训练轮数
};


//多分类器只是个壳子，内部包含 10 个二分类器，每个二分类器负责判断"是不是某个数字"


// 多分类：10 个 One-vs-Rest 分类器
class LogisticMulti {
public:
    LogisticMulti(double lr = 0.1, int epochs = 20);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    vector<int> predict(const vector<FeatureVec>& X) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    LogisticBinary classifiers_[10];
};