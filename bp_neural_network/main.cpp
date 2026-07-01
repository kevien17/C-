#include "nn.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
using namespace std;

// ====== 评测数据结构 ======
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

// ====== NeuralNetwork ======

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size,
                              double lr, int batch_size, int epochs)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size),
      lr_(lr), batch_size_(batch_size), epochs_(epochs) {}

double NeuralNetwork::sigmoid(double z) const {
    if (z > 20) return 1.0;
    if (z < -20) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

double NeuralNetwork::sigmoid_derivative(double z) const {
    // z 已经是 sigmoid 的输出：sigmoid'(z) = z * (1 - z)
    return z * (1.0 - z);
}

void NeuralNetwork::softmax(vector<double>& z) const {
    // z 是原始分数，原地转成概率
    double max_z = *max_element(z.begin(), z.end());
    double sum = 0.0;
    for (int i = 0; i < output_size_; i++) {
        z[i] = exp(z[i] - max_z);  // 减最大值防 exp 爆炸
        sum += z[i];
    }
    for (int i = 0; i < output_size_; i++) z[i] /= sum;
}

// ---- 前向传播 ----
void NeuralNetwork::forward(const FeatureVec& x,
                             vector<double>& h,      // 隐藏层输出 (128 维)
                             vector<double>& out) const {  // 输出层概率 (10 维)
    // 第1层：输入 → 隐藏
    // h_j = sigmoid( sum_i x_i * W1[i][j] + b1[j] )
    h.resize(hidden_size_);
    for (int j = 0; j < hidden_size_; j++) {
        double z = b1_[j];
        for (int i = 0; i < input_size_; i++) z += x[i] * W1_[i][j];
        h[j] = sigmoid(z);
    }

    // 第2层：隐藏 → 输出
    // out_k = sum_j h_j * W2[j][k] + b2[k] → softmax
    out.resize(output_size_);
    for (int k = 0; k < output_size_; k++) {
        double z = b2_[k];
        for (int j = 0; j < hidden_size_; j++) z += h[j] * W2_[j][k];
        out[k] = z;  // 先存原始分数，后面调 softmax 统一转
    }
    softmax(out);
}

// ---- 反向传播 ----
void NeuralNetwork::backward(const FeatureVec& x,
                              const vector<double>& y_true,
                              const vector<double>& h,
                              const vector<double>& out,
                              vector<vector<double>>& dW1,
                              vector<double>& db1,
                              vector<vector<double>>& dW2,
                              vector<double>& db2) {
    // 输出层误差: delta2 = out - y_true
    // (交叉熵 + softmax 合起来的导数就是这个，推导复杂但结论简单)
    vector<double> delta2(output_size_);
    for (int k = 0; k < output_size_; k++) {
        delta2[k] = out[k] - y_true[k];
        // dW2[j][k] = h[j] * delta2[k]
        for (int j = 0; j < hidden_size_; j++) {
            dW2[j][k] += h[j] * delta2[k];
        }
        db2[k] += delta2[k];
    }

    // 隐藏层误差: delta1_j = (sum_k delta2_k * W2[j][k]) * sigmoid'(h_j)
    vector<double> delta1(hidden_size_);
    for (int j = 0; j < hidden_size_; j++) {
        double error = 0.0;
        for (int k = 0; k < output_size_; k++) {
            error += delta2[k] * W2_[j][k];
        }
        delta1[j] = error * sigmoid_derivative(h[j]);
        // dW1[i][j] = x[i] * delta1[j]
        for (int i = 0; i < input_size_; i++) {
            dW1[i][j] += x[i] * delta1[j];
        }
        db1[j] += delta1[j];
    }
}

// ---- 训练 ----
void NeuralNetwork::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    int N = X.size();
    mt19937 rng(42);

    // 初始化权重：Xavier 初始化
    W1_.assign(input_size_, vector<double>(hidden_size_));
    normal_distribution<> dist_W1(0.0, sqrt(2.0 / input_size_));
    for (int i = 0; i < input_size_; i++)
        for (int j = 0; j < hidden_size_; j++)
            W1_[i][j] = dist_W1(rng);

    b1_.assign(hidden_size_, 0.0);

    W2_.assign(hidden_size_, vector<double>(output_size_));
    normal_distribution<> dist_W2(0.0, sqrt(2.0 / hidden_size_));
    for (int j = 0; j < hidden_size_; j++)
        for (int k = 0; k < output_size_; k++)
            W2_[j][k] = dist_W2(rng);

    b2_.assign(output_size_, 0.0);

    // 训练循环
    for (int epoch = 0; epoch < epochs_; epoch++) {
        // 打乱数据
        vector<int> idx(N);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);

        int num_batches = (N + batch_size_ - 1) / batch_size_;

        for (int b = 0; b < num_batches; b++) {
            // 累加梯度
            vector<vector<double>> dW1(input_size_, vector<double>(hidden_size_, 0.0));
            vector<double> db1(hidden_size_, 0.0);
            vector<vector<double>> dW2(hidden_size_, vector<double>(output_size_, 0.0));
            vector<double> db2(output_size_, 0.0);

            int start = b * batch_size_;
            int end = min(start + batch_size_, N);

            for (int i = start; i < end; i++) {
                int id = idx[i];

                // One-hot 标签
                vector<double> y_true(output_size_, 0.0);
                y_true[y[id]] = 1.0;

                // 前向传播
                vector<double> h, out;
                forward(X[id], h, out);

                // 反向传播，累加梯度
                backward(X[id], y_true, h, out, dW1, db1, dW2, db2);
            }

            // 取平均 + 更新参数
            int batch_n = end - start;
            double inv_n = 1.0 / batch_n;

            for (int i = 0; i < input_size_; i++)
                for (int j = 0; j < hidden_size_; j++)
                    W1_[i][j] -= lr_ * dW1[i][j] * inv_n;

            for (int j = 0; j < hidden_size_; j++)
                b1_[j] -= lr_ * db1[j] * inv_n;

            for (int j = 0; j < hidden_size_; j++)
                for (int k = 0; k < output_size_; k++)
                    W2_[j][k] -= lr_ * dW2[j][k] * inv_n;

            for (int k = 0; k < output_size_; k++)
                b2_[k] -= lr_ * db2[k] * inv_n;
        }

        // 打印进度
        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            cout << "  epoch " << epoch + 1 << "/" << epochs_ << endl;
        }
    }
}

// ---- 预测 ----
vector<int> NeuralNetwork::predict(const vector<FeatureVec>& X) const {
    auto proba = predict_proba(X);
    vector<int> preds(X.size());
    for (int i = 0; i < X.size(); i++) {
        int best = 0;
        for (int k = 1; k < output_size_; k++)
            if (proba[i][k] > proba[i][best]) best = k;
        preds[i] = best;
    }
    return preds;
}

vector<vector<double>> NeuralNetwork::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> result;
    for (const auto& x : X) {
        vector<double> h, out;
        forward(x, h, out);
        result.push_back(out);
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

// ====== Main ======
int main() {
    cout << "Loading training images..." << endl;
    auto train_data = load_images("../data/train-images-idx3-ubyte");
    auto train_labels = load_labels("../data/train-labels-idx1-ubyte");
    cout << "Loading test images..." << endl;
    auto test_data = load_images("../data/t10k-images-idx3-ubyte");
    auto test_labels = load_labels("../data/t10k-labels-idx1-ubyte");
    cout << "Train: " << train_data.num_samples << " samples" << endl;
    cout << "Test:  " << test_data.num_samples << " samples" << endl;

    int train_n = 5000;
    int test_n = 1000;
    vector<FeatureVec> train_X(train_data.images.begin(), train_data.images.begin() + train_n);
    vector<int> train_y(train_labels.begin(), train_labels.begin() + train_n);
    vector<FeatureVec> test_X(test_data.images.begin(), test_data.images.begin() + test_n);
    vector<int> test_y(test_labels.begin(), test_labels.begin() + test_n);

    // 784→128→10, lr=0.1, batch=64, epochs=20
    NeuralNetwork nn(784, 128, 10, 0.1, 64, 20);
    nn.fit(train_X, train_y);

    cout << "\nPredicting..." << endl;
    auto preds = nn.predict(test_X);
    auto proba = nn.predict_proba(test_X);
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
