#pragma once
#include "types.h"

// 决策树节点
struct TreeNode {
    int feature = -1;       // 用哪个像素分割（-1 表示叶子）
    double threshold = 0.0; // 分割阈值
    int left = -1;          // 左子节点索引
    int right = -1;         // 右子节点索引
    int label = 0;          // 预测类别（叶子节点用）
    vector<double> proba;   // 各类别概率（叶子节点用）
};

// 单棵决策树
class DecisionTree {
public:
    DecisionTree(int max_depth = 15, int min_samples = 5, int max_features = 28, int seed = 42);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    int predict_one(const FeatureVec& x) const;
    vector<int> predict(const vector<FeatureVec>& X) const;
    vector<double> predict_proba_one(const FeatureVec& x) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    double gini(const vector<int>& subset_idx, const vector<int>& y) const;
    int majority_vote(const vector<int>& subset_idx, const vector<int>& y) const;
    vector<double> class_proba(const vector<int>& subset_idx, const vector<int>& y) const;
    void build_tree(const vector<FeatureVec>& X, const vector<int>& y,
                    vector<int>& subset_idx, int depth);

    int max_depth_;
    int min_samples_;
    int max_features_;
    int seed_;
    int num_classes_ = 10;
    vector<TreeNode> nodes_;
};

// 随机森林
class RandomForest {
public:
    RandomForest(int n_trees = 20, int max_depth = 15, int min_samples = 5, int max_features = 28);

    void fit(const vector<FeatureVec>& X, const vector<int>& y);
    vector<int> predict(const vector<FeatureVec>& X) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    int n_trees_;
    int max_depth_;
    int min_samples_;
    int max_features_;
    vector<DecisionTree> trees_;
};
