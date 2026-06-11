#include "logistic.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "mnist.h"
#include "metrics.h"
using namespace std;

// ====== 数据加载 ======
// [从 knn/main.cpp 搬过来：read_int、load_images、load_labels]


// ====== 逻辑回归二分类器 ======

LogisticBinary::LogisticBinary(double lr, int epochs)
    : lr_(lr), epochs_(epochs), b_(0.0) {}//构造函数

double LogisticBinary::sigmoid(double z) const {
    if (z > 20.0) return 1.0;      // 防止 exp 爆炸,z>20的时候 exp(-z) 就非常趋近于1了
    if (z < -20.0) return 0.0;
    return 1.0 / (1.0 + exp(-z));//exp 是 e 的幂函数
}

void LogisticBinary::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    int N = X.size();// 样本数量（二维数组的行数）
    int D = X[0].size();  // 特征数量（每张图片 28*28=784）（二维数组的列数）

    // 初始化权重：用小的随机数
    mt19937 rng(42);  // 固定种子，结果可复现
    normal_distribution<> dist(0.0, 0.01);// 生成均值 0，标准差 0.01 的正态分布随机数
    w_.resize(D);
    for (int i = 0; i < D; i++) w_[i] = dist(rng);
    b_ = 0.0;

    // 梯度下降
    for (int epoch = 0; epoch < epochs_; epoch++) {
        // 打乱数据顺序
        vector<int> idx(N);
        for (int i = 0; i < N; i++) idx[i] = i;
        shuffle(idx.begin(), idx.end(), rng);//打乱索引顺序，避免梯度方向卡在局部最低点，帮助模型更好地收敛

        // 遍历所有样本
        for (int i = 0; i < N; i++) {
            int id = idx[i];

            // 1. 算 z = w·x + b
            double z = b_;
            const auto& x = X[id];
            for (int j = 0; j < D; j++) {
                z += w_[j] * x[j];
            }

            // 2. 算误差
            double error = sigmoid(z) - y[id];

            // 3. 更新权重和偏置（沿梯度反方向走）
            for (int j = 0; j < D; j++) {
                w_[j] -= lr_ * error * x[j];
            }
            b_ -= lr_ * error;
        }

        // 打印训练进度
        cout << "  epoch " << epoch + 1 << "/" << epochs_ << endl;
    }
}

vector<double> LogisticBinary::predict_proba(const vector<FeatureVec>& X) const {
    vector<double> result;
    int D = w_.size();
    for (const auto& x : X) {
        double z = b_;
        for (int j = 0; j < D; j++) {
            z += w_[j] * x[j];
        }
        result.push_back(sigmoid(z));
    }
    return result;
}

// ====== 多分类器 ======

LogisticMulti::LogisticMulti(double lr, int epochs) {
    for (int c = 0; c < 10; c++) {
        classifiers_[c] = LogisticBinary(lr, epochs);
    }
}

void LogisticMulti::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    for (int c = 0; c < 10; c++) {
        cout << "训练分类器: 数字 " << c << " vs 其他" << endl;

        // 造二分类标签：是 c → 1，不是 c → 0
        vector<int> binary_y(y.size());
        for (int i = 0; i < y.size(); i++) {
            binary_y[i] = (y[i] == c) ? 1 : 0;
        }

        classifiers_[c].fit(X, binary_y);
    }
}

vector<int> LogisticMulti::predict(const vector<FeatureVec>& X) const {
    auto proba = predict_proba(X);
    vector<int> preds(X.size());
    for (int i = 0; i < X.size(); i++) {
        int best = 0;
        for (int c = 1; c < 10; c++) {
            if (proba[i][c] > proba[i][best]) best = c;
        }
        preds[i] = best;
    }
    return preds;
}

vector<vector<double>> LogisticMulti::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> proba(X.size(), vector<double>(10));
    for (int c = 0; c < 10; c++) {
        auto scores = classifiers_[c].predict_proba(X);
        for (int i = 0; i < X.size(); i++) {
            proba[i][c] = scores[i];
        }
    }
    return proba;
}




//————————————————————————————————————————————————数据加载部分——————————————————————————————————————————————————————————————





static int read_int(ifstream& f) {
    unsigned char buf[4];
    f.read((char*)buf, 4);
    // 大端序：第一个字节是最高位
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}
Dataset load_images(const string& filepath) {
    Dataset data;
    ifstream f(filepath, ios::binary);
    if (!f) {
        cout << "打不开文件: " << filepath << endl;
        return data;
    }

    int magic = read_int(f);
    int count = read_int(f);
    int rows  = read_int(f);
    int cols  = read_int(f);

    // 验一下身份证
    if (magic != 2051) {
        cout << "文件 magic 不对: " << magic << " (应该是 2051)" << endl;
        return data;
    }

    data.rows = rows;
    data.cols = cols;
    data.num_samples = count;

    // 逐张读取图片
    for (int i = 0; i < count; i++) {
        FeatureVec img(rows * cols);  // 784 个 double
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            f.read((char*)&pixel, 1);
            img[j] = pixel / 255.0;   // 归一化到 [0, 1]
        }
        data.images.push_back(img);
    }

    return data;
}
vector<int> load_labels(const string& filepath) {
    vector<int> labels;
    ifstream f(filepath, ios::binary);
    if (!f) {
        cout << "打不开文件: " << filepath << endl;
        return labels;
    }

    int magic = read_int(f);
    int count = read_int(f);

    if (magic != 2049) {
        cout << "文件 magic 不对: " << magic << " (应该是 2049)" << endl;
        return labels;
    }

    for (int i = 0; i < count; i++) {
        unsigned char label;
        f.read((char*)&label, 1);
        labels.push_back((int)label);
    }

    return labels;
}






//————————————————————————————————————————————————评测指标部分——————————————————————————————————————————————————————————————





vector<pair<double, double>> compute_roc(
    const vector<int>& true_labels,
    const vector<double>& scores)
{
    int N = scores.size();

    // 数一下正样本有多少
    int P = 0;
    for (int i = 0; i < N; i++) {
        if (true_labels[i] == 1) P++;
    }
    int Neg = N - P;

    // 全是同一类就别画了
    if (P == 0 || Neg == 0) {
        return {{0.0, 0.0}, {1.0, 1.0}};
    }

    // 按预测分数从大到小排序（高分的先判为正类）
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);    // 0,1,2,...,N-1
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    vector<pair<double, double>> roc;
    roc.push_back({0.0, 0.0});

    int TP = 0, FP = 0;
    for (int i = 0; i < N; i++) {
        if (true_labels[idx[i]] == 1)
            TP++;
        else
            FP++;

        double tpr = (double)TP / P;          // 真阳性率
        double fpr = (double)FP / Neg;        // 假阳性率
        roc.push_back({fpr, tpr});
    }

    return roc;
}

// ---------- 算 AUC（梯形法）----------
double compute_auc(const vector<pair<double, double>>& roc) {
    double area = 0.0;
    for (int i = 1; i < roc.size(); i++) {
        double width = roc[i].first - roc[i - 1].first;
        double avg_height = (roc[i].second + roc[i - 1].second) / 2.0;
        area += width * avg_height;
    }
    return area;
}

// ---------- 完整评测 ----------
EvalResult evaluate(
    const vector<int>& true_labels,
    const vector<int>& predicted,
    const vector<vector<double>>& proba,
    int num_classes)
{
    EvalResult result;
    int N = true_labels.size();

    // 1. Accuracy
    int correct = 0;
    for (int i = 0; i < N; i++) {
        if (true_labels[i] == predicted[i]) correct++;
    }
    result.accuracy = (double)correct / N;

    // 2. 每个类别算 precision / recall / F1 / AUC / ROC
    result.precision.resize(num_classes);
    result.recall.resize(num_classes);
    result.f1.resize(num_classes);
    result.auc.resize(num_classes);
    result.roc_curves.resize(num_classes);

    for (int k = 0; k < num_classes; k++) {
        // 把多分类转成二分类："是不是 k？"
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
                                / (result.precision[k] + result.recall[k])
                              : 0.0;

        // 造二分类标签和分数
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

























// ————————————————————————————————————————————————主函数部分——————————————————————————————————————————————————————————————

int main() {
    // 1. 加载数据
    cout << "加载训练图片..." << endl;
    auto train_data = load_images("../data/train-images-idx3-ubyte");
    auto train_labels = load_labels("../data/train-labels-idx1-ubyte");

    cout << "加载测试图片..." << endl;
    auto test_data = load_images("../data/t10k-images-idx3-ubyte");
    auto test_labels = load_labels("../data/t10k-labels-idx1-ubyte");

    cout << "训练集: " << train_data.num_samples << " 张" << endl;
    cout << "测试集: " << test_data.num_samples << " 张" << endl;

    // 2. 先用 2000 张训练、500 张测试（逻辑回归比 KNN 快多了，但先小规模跑通）
    int train_n = 2000;
    int test_n = 500;

    vector<FeatureVec> train_X(train_data.images.begin(),
                               train_data.images.begin() + train_n);
    vector<int> train_y(train_labels.begin(),
                        train_labels.begin() + train_n);

    vector<FeatureVec> test_X(test_data.images.begin(),
                              test_data.images.begin() + test_n);
    vector<int> test_y(test_labels.begin(),
                       test_labels.begin() + test_n);

    // 3. 训练
    LogisticMulti model(0.1, 20);  // 学习率 0.1，20 轮
    model.fit(train_X, train_y);

    // 4. 预测
    cout << "\n开始预测..." << endl;
    auto preds = model.predict(test_X);

       // 4. 完整评测
    auto proba = model.predict_proba(test_X);
    auto result = evaluate(test_y, preds, proba, 10);

    cout << "\n========== 评测结果 ==========\n" << endl;
    cout << "Accuracy: " << result.accuracy * 100 << "%" << endl;

    cout << "\n数字  Precision  Recall  F1      AUC" << endl;
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