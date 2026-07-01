#include "random_forest.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
using namespace std;

// —————————————————————————————————— 评测数据结构 ————————————————————————————————————————
struct PerClassMetrics { double precision, recall, f1, auc; };
struct EvalResult {
    double accuracy;
    vector<double> precision, recall, f1, auc;
    vector<vector<pair<double, double>>> roc_curves;
};

// ====== 数据加载 ======
static int read_int(ifstream& f) {
    unsigned char buf[4];
    f.read((char*)buf, 4);
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}
Dataset load_images(const string& filepath) {
    Dataset data;
    ifstream f(filepath, ios::binary);
    if (!f) { cout << "Cannot open: " << filepath << endl; return data; }
    int magic = read_int(f), count = read_int(f), rows = read_int(f), cols = read_int(f);
    data.rows = rows; data.cols = cols; data.num_samples = count;
    for (int i = 0; i < count; i++) {
        FeatureVec img(rows * cols);
        for (int j = 0; j < rows * cols; j++) {
            unsigned char p; f.read((char*)&p, 1);
            img[j] = p / 255.0;
        }
        data.images.push_back(img);
    }
    return data;
}
vector<int> load_labels(const string& filepath) {
    vector<int> labels;
    ifstream f(filepath, ios::binary);
    if (!f) return labels;
    int magic = read_int(f), count = read_int(f);
    for (int i = 0; i < count; i++) {
        unsigned char l; f.read((char*)&l, 1);
        labels.push_back(l);
    }
    return labels;
}













// —————————————————————————————————— DecisionTree ————————————————————————————————————————

DecisionTree::DecisionTree(int max_depth, int min_samples, int max_features, int seed)
    : max_depth_(max_depth), min_samples_(min_samples),
      max_features_(max_features), seed_(seed) {}

// Gini impurity: 衡量一堆样本的"混乱程度"
// Gini = 1 - sum(p_k^2), p_k 是第 k 类的比例
double DecisionTree::gini(const vector<int>& subset_idx, const vector<int>& y) const {
    if (subset_idx.empty()) return 0.0;
    double counts[10] = {0};
    for (int idx : subset_idx) counts[y[idx]]++;
    double sum_sq = 0.0;
    for (int k = 0; k < num_classes_; k++) {
        double p = counts[k] / subset_idx.size();
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

int DecisionTree::majority_vote(const vector<int>& subset_idx, const vector<int>& y) const {
    int counts[10] = {0};
    for (int idx : subset_idx) counts[y[idx]]++;
    int best = 0;
    for (int k = 1; k < num_classes_; k++)
        if (counts[k] > counts[best]) best = k;
    return best;
}

vector<double> DecisionTree::class_proba(const vector<int>& subset_idx, const vector<int>& y) const {
    vector<double> proba(num_classes_, 0.0);
    for (int idx : subset_idx) proba[y[idx]]++;
    for (int k = 0; k < num_classes_; k++) proba[k] /= subset_idx.size();
    return proba;
}

void DecisionTree::build_tree(const vector<FeatureVec>& X, const vector<int>& y,
                               vector<int>& subset_idx, int depth) {
    int N = subset_idx.size();

    // 停止条件：样本太少 或 太纯 或 深度够了
    double current_gini = gini(subset_idx, y);
    if (N < min_samples_ * 2 || depth >= max_depth_ || current_gini < 0.001) {
        TreeNode leaf;
        leaf.label = majority_vote(subset_idx, y);
        leaf.proba = class_proba(subset_idx, y);
        nodes_.push_back(leaf);
        return;
    }

    int D = X[0].size();
    mt19937 rng(seed_ + depth + nodes_.size());  // 随机种子 = 树种子 + 深度 + 节点数

    // 随机选 max_features_ 个像素
    vector<int> candidate_features(D);
    iota(candidate_features.begin(), candidate_features.end(), 0);
    shuffle(candidate_features.begin(), candidate_features.end(), rng);
    candidate_features.resize(max_features_);

    // 遍历每个候选特征，找最佳阈值
    double best_gini = 1e9;
    int best_feature = -1;
    double best_threshold = 0.0;
    vector<int> best_left, best_right;

    for (int feat : candidate_features) {
        // 收集该特征在当前子集上的所有值
        vector<double> values;
        for (int idx : subset_idx) values.push_back(X[idx][feat]);
        sort(values.begin(), values.end());

        // 尝试每个可能的阈值（相邻值的中间点）
        for (int t = 0; t < (int)values.size() - 1; t++) {
            double threshold = (values[t] + values[t + 1]) / 2.0;

            // 按阈值分成左右两拨
            vector<int> left_sub, right_sub;
            for (int idx : subset_idx) {
                if (X[idx][feat] <= threshold) left_sub.push_back(idx);
                else right_sub.push_back(idx);
            }
            if (left_sub.empty() || right_sub.empty()) continue;

            // 计算加权 Gini
            double left_gini = gini(left_sub, y);
            double right_gini = gini(right_sub, y);
            double weighted = (left_gini * left_sub.size() + right_gini * right_sub.size()) / N;

            if (weighted < best_gini) {
                best_gini = weighted;
                best_feature = feat;
                best_threshold = threshold;
                best_left = move(left_sub);
                best_right = move(right_sub);
            }
        }
    }

    // 找不到好的分割 → 变叶子
    if (best_feature == -1) {
        TreeNode leaf;
        leaf.label = majority_vote(subset_idx, y);
        leaf.proba = class_proba(subset_idx, y);
        nodes_.push_back(leaf);
        return;
    }

    // 记录当前节点
    TreeNode node;
    node.feature = best_feature;
    node.threshold = best_threshold;
    int current_idx = nodes_.size();
    nodes_.push_back(node);

    // 递归建左右子树
    int left_idx = nodes_.size();
    build_tree(X, y, best_left, depth + 1);

    int right_idx = nodes_.size();
    build_tree(X, y, best_right, depth + 1);

    // 回填子节点索引（左右建完后才能知道位置）
    nodes_[current_idx].left = left_idx;
    nodes_[current_idx].right = right_idx;
}

void DecisionTree::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    // Bootstrap：有放回随机抽样，每棵树看到 ~63% 的原始样本
    int N = X.size();
    mt19937 rng(seed_);  // 每棵树不同的随机种子
    vector<int> subset_idx(N);
    for (int i = 0; i < N; i++)
        subset_idx[i] = rng() % N;  // 有放回随机选

    build_tree(X, y, subset_idx, 0);
}

int DecisionTree::predict_one(const FeatureVec& x) const {
    int node_idx = 0;
    while (nodes_[node_idx].feature != -1) {
        if (x[nodes_[node_idx].feature] <= nodes_[node_idx].threshold)
            node_idx = nodes_[node_idx].left;
        else
            node_idx = nodes_[node_idx].right;
    }
    return nodes_[node_idx].label;
}

vector<int> DecisionTree::predict(const vector<FeatureVec>& X) const {
    vector<int> result;
    for (const auto& x : X) result.push_back(predict_one(x));
    return result;
}

vector<double> DecisionTree::predict_proba_one(const FeatureVec& x) const {
    int node_idx = 0;
    while (nodes_[node_idx].feature != -1) {
        if (x[nodes_[node_idx].feature] <= nodes_[node_idx].threshold)
            node_idx = nodes_[node_idx].left;
        else
            node_idx = nodes_[node_idx].right;
    }
    return nodes_[node_idx].proba;
}

vector<vector<double>> DecisionTree::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> result;
    for (const auto& x : X) result.push_back(predict_proba_one(x));
    return result;
}














// ———————————————————————————————————————— RandomForest —————————————————————————————————————————

RandomForest::RandomForest(int n_trees, int max_depth, int min_samples, int max_features)
    : n_trees_(n_trees), max_depth_(max_depth),
      min_samples_(min_samples), max_features_(max_features) {}

void RandomForest::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    trees_.clear();
    for (int t = 0; t < n_trees_; t++) {
        cout << "Training tree " << t + 1 << "/" << n_trees_ << "..." << endl;
        DecisionTree tree(max_depth_, min_samples_, max_features_, t * 100 + 42);
        tree.fit(X, y);
        trees_.push_back(move(tree));
    }
}

vector<int> RandomForest::predict(const vector<FeatureVec>& X) const {
    vector<int> result;
    for (int i = 0; i < X.size(); i++) {
        int votes[10] = {0};
        for (const auto& tree : trees_) {
            votes[tree.predict_one(X[i])]++;
        }
        int best = 0;
        for (int k = 1; k < 10; k++)
            if (votes[k] > votes[best]) best = k;
        result.push_back(best);
    }
    return result;
}

vector<vector<double>> RandomForest::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> result;
    for (int i = 0; i < X.size(); i++) {
        vector<double> proba(10, 0.0);
        for (const auto& tree : trees_) {
            auto tree_proba = tree.predict_proba_one(X[i]);
            for (int k = 0; k < 10; k++) proba[k] += tree_proba[k];
        }
        for (int k = 0; k < 10; k++) proba[k] /= n_trees_;
        result.push_back(proba);
    }
    return result;
}

// ====== Evaluation ======
vector<pair<double, double>> compute_roc(
    const vector<int>& true_labels, const vector<double>& scores)
{
    int N = scores.size();
    int P = 0;
    for (int i = 0; i < N; i++) if (true_labels[i] == 1) P++;
    int Neg = N - P;
    if (P == 0 || Neg == 0) return {{0.0, 0.0}, {1.0, 1.0}};

    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    vector<pair<double, double>> roc;
    roc.push_back({0.0, 0.0});
    int TP = 0, FP = 0;
    for (int i = 0; i < N; i++) {
        if (true_labels[idx[i]] == 1) TP++; else FP++;
        roc.push_back({(double)FP / Neg, (double)TP / P});
    }
    return roc;
}

double compute_auc(const vector<pair<double, double>>& roc) {
    double area = 0.0;
    for (int i = 1; i < roc.size(); i++) {
        double w = roc[i].first - roc[i - 1].first;
        double h = (roc[i].second + roc[i - 1].second) / 2.0;
        area += w * h;
    }
    return area;
}

EvalResult evaluate(const vector<int>& true_labels, const vector<int>& predicted,
                    const vector<vector<double>>& proba, int num_classes)
{
    EvalResult result;
    int N = true_labels.size();
    int correct = 0;
    for (int i = 0; i < N; i++)
        if (true_labels[i] == predicted[i]) correct++;
    result.accuracy = (double)correct / N;

    result.precision.resize(num_classes);
    result.recall.resize(num_classes);
    result.f1.resize(num_classes);
    result.auc.resize(num_classes);
    result.roc_curves.resize(num_classes);

    for (int k = 0; k < num_classes; k++) {
        int TP = 0, FP = 0, FN = 0;
        for (int i = 0; i < N; i++) {
            if (true_labels[i] == k && predicted[i] == k) TP++;
            if (true_labels[i] != k && predicted[i] == k) FP++;
            if (true_labels[i] == k && predicted[i] != k) FN++;
        }
        result.precision[k] = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
        result.recall[k]    = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
        result.f1[k]        = (result.precision[k] + result.recall[k] > 0)
                              ? 2 * result.precision[k] * result.recall[k]
                                / (result.precision[k] + result.recall[k]) : 0.0;

        vector<int> binary_y(N);
        vector<double> scores(N);
        for (int i = 0; i < N; i++) {
            binary_y[i] = (true_labels[i] == k) ? 1 : 0;
            scores[i] = proba[i][k];
        }
        result.roc_curves[k] = compute_roc(binary_y, scores);
        result.auc[k] = compute_auc(result.roc_curves[k]);
    }
    return result;
}








//———————————————————————————————————————— Main ————————————————————————————————————————
int main() {
    cout << "Loading training images..." << endl;
    auto train_data = load_images("../data/train-images-idx3-ubyte");
    auto train_labels = load_labels("../data/train-labels-idx1-ubyte");
    cout << "Loading test images..." << endl;
    auto test_data = load_images("../data/t10k-images-idx3-ubyte");
    auto test_labels = load_labels("../data/t10k-labels-idx1-ubyte");
    cout << "Train: " << train_data.num_samples << " samples" << endl;
    cout << "Test:  " << test_data.num_samples << " samples" << endl;

    int train_n = 2000;
    int test_n = 500;
    vector<FeatureVec> train_X(train_data.images.begin(), train_data.images.begin() + train_n);
    vector<int> train_y(train_labels.begin(), train_labels.begin() + train_n);
    vector<FeatureVec> test_X(test_data.images.begin(), test_data.images.begin() + test_n);
    vector<int> test_y(test_labels.begin(), test_labels.begin() + test_n);

    // 20棵树，max_depth=15，min_samples=5，max_features=28 (sqrt(784)=28)
    RandomForest forest(20, 15, 5, 28);
    forest.fit(train_X, train_y);

    cout << "\nPredicting..." << endl;
    auto preds = forest.predict(test_X);
    auto proba = forest.predict_proba(test_X);
    auto result = evaluate(test_y, preds, proba, 10);

    cout << "\nAccuracy: " << result.accuracy * 100 << "%" << endl;
    cout << "\nDigit Precision Recall F1      AUC" << endl;
    cout << "----------------------------------------" << endl;
    for (int k = 0; k < 10; k++) {
        cout << " " << k << "    "
             << result.precision[k] << "     "
             << result.recall[k] << "     "
             << result.f1[k] << "     "
             << result.auc[k] << endl;
    }
    return 0;
}
