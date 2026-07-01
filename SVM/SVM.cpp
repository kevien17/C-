#include "svm.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
using namespace std;

// ====== 评测数据结构 ======
struct PerClassMetrics {
    double precision, recall, f1, auc;
};
struct EvalResult {
    double accuracy;
    vector<double> precision, recall, f1, auc;
    double macro_precision, macro_recall, macro_f1, macro_auc;
    vector<vector<pair<double, double>>> roc_curves;
    vector<vector<int>> confusion;
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

// ====== SVM Binary ======
SVMBinary::SVMBinary(double lr, double lambda, int epochs)
    : lr_(lr), lambda_(lambda), epochs_(epochs), b_(0.0) {}

void SVMBinary::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    int N = X.size(), D = X[0].size();
    mt19937 rng(42);
    normal_distribution<> dist(0.0, 0.01);
    w_.resize(D);
    for (int i = 0; i < D; i++) w_[i] = dist(rng);
    b_ = 0.0;

    for (int epoch = 0; epoch < epochs_; epoch++) {
        vector<int> idx(N);
        for (int i = 0; i < N; i++) idx[i] = i;
        shuffle(idx.begin(), idx.end(), rng);

        for (int i = 0; i < N; i++) {
            int id = idx[i];
            double z = b_;
            const auto& x = X[id];
            for (int j = 0; j < D; j++) z += w_[j] * x[j];

            // Hinge loss: y in {-1, +1}
            int y_svm = (y[id] == 1) ? 1 : -1;
            double hinge = 1.0 - y_svm * z;

            if (hinge > 0) {
                double grad = -y_svm;
                for (int j = 0; j < D; j++)
                    w_[j] -= lr_ * (grad * x[j] + 2.0 * lambda_ * w_[j]);
                b_ -= lr_ * grad;
            } else {
                for (int j = 0; j < D; j++)
                    w_[j] -= lr_ * 2.0 * lambda_ * w_[j];
            }
        }
        cout << "  epoch " << epoch + 1 << "/" << epochs_ << endl;
    }
}

vector<double> SVMBinary::predict_proba(const vector<FeatureVec>& X) const {
    int D = w_.size();
    vector<double> result;
    for (const auto& x : X) {
        double z = b_;
        for (int j = 0; j < D; j++) z += w_[j] * x[j];
        if (z > 20) result.push_back(1.0);
        else if (z < -20) result.push_back(0.0);
        else result.push_back(1.0 / (1.0 + exp(-z)));
    }
    return result;
}

// ====== SVM Multi-class ======
SVMMulti::SVMMulti(double lr, double lambda, int epochs) {
    for (int c = 0; c < 10; c++) classifiers_[c] = SVMBinary(lr, lambda, epochs);
}
void SVMMulti::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    for (int c = 0; c < 10; c++) {
        cout << "Training SVM for digit " << c << " vs rest..." << endl;
        vector<int> binary_y(y.size());
        for (int i = 0; i < y.size(); i++) binary_y[i] = (y[i] == c) ? 1 : 0;
        classifiers_[c].fit(X, binary_y);
    }
}
vector<int> SVMMulti::predict(const vector<FeatureVec>& X) const {
    auto proba = predict_proba(X);
    vector<int> preds(X.size());
    for (int i = 0; i < X.size(); i++) {
        int best = 0;
        for (int c = 1; c < 10; c++)
            if (proba[i][c] > proba[i][best]) best = c;
        preds[i] = best;
    }
    return preds;
}
vector<vector<double>> SVMMulti::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> proba(X.size(), vector<double>(10));
    for (int c = 0; c < 10; c++) {
        auto scores = classifiers_[c].predict_proba(X);
        for (int i = 0; i < X.size(); i++) proba[i][c] = scores[i];
    }
    return proba;
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

    int train_n = 2000;
    int test_n = 500;
    vector<FeatureVec> train_X(train_data.images.begin(), train_data.images.begin() + train_n);
    vector<int> train_y(train_labels.begin(), train_labels.begin() + train_n);
    vector<FeatureVec> test_X(test_data.images.begin(), test_data.images.begin() + test_n);
    vector<int> test_y(test_labels.begin(), test_labels.begin() + test_n);

    SVMMulti model(0.01, 0.001, 20);
    model.fit(train_X, train_y);

    cout << "\nPredicting..." << endl;
    auto preds = model.predict(test_X);
    auto proba = model.predict_proba(test_X);
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
