#pragma once
#include "types.h"

// 二分类 SVM：判断"是不是某个数字"
class SVMBinary {
public:
    SVMBinary(double lr = 0.01, double lambda = 0.001, int epochs = 20);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    vector<double> predict_proba(const vector<FeatureVec>& X) const;

private:
    vector<double> w_;
    double b_;
    double lr_;
    double lambda_;  // 正则化强度
    int epochs_;
};

// 多分类 SVM：10 个 One-vs-Rest
class SVMMulti {
public:
    SVMMulti(double lr = 0.01, double lambda = 0.001, int epochs = 20);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    vector<int> predict(const vector<FeatureVec>& X) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    SVMBinary classifiers_[10];
};