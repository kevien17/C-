#pragma once
#include <vector>
using namespace std;

// 算 ROC 曲线上的点
vector<pair<double, double>> compute_roc(
    const vector<int>& true_labels,  // 真实标签（0/1，二分类）
    const vector<double>& scores     // 预测概率
);

// 算 AUC
double compute_auc(const vector<pair<double, double>>& roc);

// 完整评测：accuracy + 每个类别的 precision/recall/F1/AUC + ROC
struct EvalResult {
    double accuracy;
    vector<double> precision;  // 10 个类别各一个
    vector<double> recall;
    vector<double> f1;
    vector<double> auc;
    vector<vector<pair<double, double>>> roc_curves; // 10 条 ROC 曲线
};

EvalResult evaluate(
    const vector<int>& true_labels,
    const vector<int>& predicted,
    const vector<vector<double>>& proba,  // predict_proba 的结果
    int num_classes
);