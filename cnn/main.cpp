#include "cnn.h"
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

// ====== 辅助函数 ======
// 784→28×28×1
vector<vector<vector<double>>> CNN::to_image(const FeatureVec& x) const {
    vector<vector<vector<double>>> img(1, vector<vector<double>>(28, vector<double>(28)));
    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++)
            img[0][r][c] = x[r * 28 + c];
    return img;
}

// ====== 前向传播组件 ======

void CNN::conv2d(const vector<vector<vector<double>>>& input,
                 const vector<vector<vector<vector<double>>>>& filters,
                 const vector<double>& bias,
                 vector<vector<vector<double>>>& output) const {
    int C_in = input.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int C_out = filters.size();
    int K = filters[0][0].size(); // kernel size

    output.assign(C_out, vector<vector<double>>(H, vector<double>(W, 0.0)));

    for (int co = 0; co < C_out; co++) {
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                double sum = bias[co];
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kr = 0; kr < K; kr++) {
                        int in_r = r + kr - K/2; // pad: shift by K/2
                        int in_c0 = c - K/2;
                        if (in_r < 0 || in_r >= H) continue;
                        for (int kc = 0; kc < K; kc++) {
                            int in_c = in_c0 + kc;
                            if (in_c < 0 || in_c >= W) continue;
                            sum += input[ci][in_r][in_c] * filters[co][ci][kr][kc];
                        }
                    }
                }
                output[co][r][c] = sum;
            }
        }
    }
}

void CNN::relu(vector<vector<vector<double>>>& x) const {
    for (auto& ch : x)
        for (auto& row : ch)
            for (double& v : row)
                if (v < 0) v = 0;
}

void CNN::maxpool(const vector<vector<vector<double>>>& input,
                  vector<vector<vector<double>>>& output,
                  vector<vector<vector<vector<int>>>>& max_idx) const {
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int outH = H / 2, outW = W / 2;

    output.assign(C, vector<vector<double>>(outH, vector<double>(outW, 0.0)));
    max_idx.assign(C, vector<vector<vector<int>>>(outH, vector<vector<int>>(outW, vector<int>(2))));

    for (int ch = 0; ch < C; ch++) {
        for (int r = 0; r < outH; r++) {
            for (int c = 0; c < outW; c++) {
                double mx = -1e9;
                int mx_r = 0, mx_c = 0;
                for (int dr = 0; dr < 2; dr++) {
                    for (int dc = 0; dc < 2; dc++) {
                        double v = input[ch][r*2+dr][c*2+dc];
                        if (v > mx) { mx = v; mx_r = r*2+dr; mx_c = c*2+dc; }
                    }
                }
                output[ch][r][c] = mx;
                max_idx[ch][r][c][0] = mx_r;
                max_idx[ch][r][c][1] = mx_c;
            }
        }
    }
}

void CNN::fc_forward(const vector<double>& input,
                     const vector<vector<double>>& W,
                     const vector<double>& b,
                     vector<double>& output) const {
    int out_dim = b.size();
    output.assign(out_dim, 0.0);
    for (int o = 0; o < out_dim; o++) {
        double sum = b[o];
        for (int i = 0; i < input.size(); i++) sum += input[i] * W[i][o];
        output[o] = sum;
    }
}

void CNN::softmax(vector<double>& z) const {
    double mx = *max_element(z.begin(), z.end());
    double sum = 0.0;
    for (double& v : z) { v = exp(v - mx); sum += v; }
    for (double& v : z) v /= sum;
}

// ====== 反向传播组件 ======

void CNN::relu_backward(const vector<vector<vector<double>>>& input,
                        vector<vector<vector<double>>>& grad) const {
    for (int ch = 0; ch < grad.size(); ch++)
        for (int r = 0; r < grad[ch].size(); r++)
            for (int c = 0; c < grad[ch][r].size(); c++)
                if (input[ch][r][c] <= 0) grad[ch][r][c] = 0;
}

void CNN::maxpool_backward(const vector<vector<vector<double>>>& output_grad,
                           const vector<vector<vector<vector<int>>>>& max_idx,
                           vector<vector<vector<double>>>& input_grad) const {
    // input_grad 已分配为输入尺寸，全零。只给 max 位置传梯度。
    int C = output_grad.size();
    int outH = output_grad[0].size();
    int outW = output_grad[0][0].size();

    for (int ch = 0; ch < C; ch++) {
        for (int r = 0; r < outH; r++) {
            for (int c = 0; c < outW; c++) {
                int mx_r = max_idx[ch][r][c][0];
                int mx_c = max_idx[ch][r][c][1];
                input_grad[ch][mx_r][mx_c] = output_grad[ch][r][c];
            }
        }
    }
}

void CNN::conv2d_backward(const vector<vector<vector<double>>>& input,
                          const vector<vector<vector<double>>>& output_grad,
                          const vector<vector<vector<vector<double>>>>& filters,
                          vector<vector<vector<double>>>& input_grad,
                          vector<vector<vector<vector<double>>>>& filter_grad,
                          vector<double>& bias_grad) const {
    int C_in = input.size();
    int C_out = output_grad.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int K = filters[0][0].size();

    // input_grad 初始化为零
    input_grad.assign(C_in, vector<vector<double>>(H, vector<double>(W, 0.0)));

    for (int co = 0; co < C_out; co++) {
        bias_grad[co] = 0;
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                bias_grad[co] += output_grad[co][r][c];

        for (int ci = 0; ci < C_in; ci++) {
            for (int kr = 0; kr < K; kr++) {
                for (int kc = 0; kc < K; kc++) {
                    filter_grad[co][ci][kr][kc] = 0;
                }
            }
        }
    }

    for (int co = 0; co < C_out; co++) {
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                double dout = output_grad[co][r][c];
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kr = 0; kr < K; kr++) {
                        int in_r = r + kr - K/2;
                        if (in_r < 0 || in_r >= H) continue;
                        for (int kc = 0; kc < K; kc++) {
                            int in_c = c + kc - K/2;
                            if (in_c < 0 || in_c >= W) continue;
                            // 梯度传播到输入
                            input_grad[ci][in_r][in_c] += dout * filters[co][ci][kr][kc];
                            // 梯度对 filter 的贡献
                            filter_grad[co][ci][kr][kc] += input[ci][in_r][in_c] * dout;
                        }
                    }
                }
            }
        }
    }
}

void CNN::fc_backward(const vector<double>& input,
                      const vector<double>& output_grad,
                      const vector<vector<double>>& W,
                      vector<double>& input_grad,
                      vector<vector<double>>& W_grad,
                      vector<double>& b_grad) const {
    int D_in = input.size();
    int D_out = output_grad.size();

    for (int o = 0; o < D_out; o++) {
        b_grad[o] += output_grad[o];
        for (int i = 0; i < D_in; i++) {
            W_grad[i][o] += input[i] * output_grad[o];
        }
    }

    for (int i = 0; i < D_in; i++) {
        input_grad[i] = 0;
        for (int o = 0; o < D_out; o++) {
            input_grad[i] += W[i][o] * output_grad[o];
        }
    }
}

// ====== 构造函数 & 初始化 ======

CNN::CNN(double lr, int batch_size, int epochs)
    : lr_(lr), batch_size_(batch_size), epochs_(epochs) {}

void init_conv_filters(vector<vector<vector<vector<double>>>>& W,
                        int C_out, int C_in, int K, mt19937& rng) {
    double std = sqrt(2.0 / (C_in * K * K));
    normal_distribution<> dist(0.0, std);
    W.assign(C_out, vector<vector<vector<double>>>(C_in,
        vector<vector<double>>(K, vector<double>(K))));
    for (int co = 0; co < C_out; co++)
        for (int ci = 0; ci < C_in; ci++)
            for (int kr = 0; kr < K; kr++)
                for (int kc = 0; kc < K; kc++)
                    W[co][ci][kr][kc] = dist(rng);
}

void init_fc_weights(vector<vector<double>>& W, int D_in, int D_out, mt19937& rng) {
    double std = sqrt(2.0 / D_in);
    normal_distribution<> dist(0.0, std);
    W.assign(D_in, vector<double>(D_out));
    for (int i = 0; i < D_in; i++)
        for (int o = 0; o < D_out; o++)
            W[i][o] = dist(rng);
}

// ====== 训练 ======

void CNN::fit(const vector<FeatureVec>& X, const vector<int>& y) {
    int N = X.size();
    mt19937 rng(42);

    // 初始化所有参数
    init_conv_filters(W_conv1_, 8, 1, 3, rng);
    b_conv1_.assign(8, 0.0);
    init_conv_filters(W_conv2_, 16, 8, 3, rng);
    b_conv2_.assign(16, 0.0);
    init_fc_weights(W_fc1_, 784, 128, rng);
    b_fc1_.assign(128, 0.0);
    init_fc_weights(W_fc2_, 128, 10, rng);
    b_fc2_.assign(10, 0.0);

    for (int epoch = 0; epoch < epochs_; epoch++) {
        vector<int> idx(N);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);

        int num_batches = (N + batch_size_ - 1) / batch_size_;

        for (int b = 0; b < num_batches; b++) {
            // 累加器
            auto dW1 = W_conv1_; for (auto& a : dW1) for (auto& b2 : a) for (auto& c : b2) for (double& v : c) v = 0;
            vector<double> db1(8, 0);
            auto dW2 = W_conv2_; for (auto& a : dW2) for (auto& b2 : a) for (auto& c : b2) for (double& v : c) v = 0;
            vector<double> db2(16, 0);
            vector<vector<double>> dW_fc1(784, vector<double>(128, 0));
            vector<double> db_fc1(128, 0);
            vector<vector<double>> dW_fc2(128, vector<double>(10, 0));
            vector<double> db_fc2(10, 0);

            int start = b * batch_size_;
            int end = min(start + batch_size_, N);
            int batch_n = end - start;

            for (int i = start; i < end; i++) {
                int id = idx[i];

                // ==== 前向传播 ====
                auto img = to_image(X[id]);

                // Conv1 + ReLU
                vector<vector<vector<double>>> conv1_out;
                conv2d(img, W_conv1_, b_conv1_, conv1_out);
                relu(conv1_out);

                // Pool1
                vector<vector<vector<double>>> pool1_out;
                vector<vector<vector<vector<int>>>> max_idx1;
                maxpool(conv1_out, pool1_out, max_idx1);

                // Conv2 + ReLU
                vector<vector<vector<double>>> conv2_out;
                conv2d(pool1_out, W_conv2_, b_conv2_, conv2_out);
                relu(conv2_out);

                // Pool2
                vector<vector<vector<double>>> pool2_out;
                vector<vector<vector<vector<int>>>> max_idx2;
                maxpool(conv2_out, pool2_out, max_idx2);

                // Flatten: [16][7][7] → 784
                vector<double> flat(784);
                for (int ch = 0; ch < 16; ch++)
                    for (int r = 0; r < 7; r++)
                        for (int c = 0; c < 7; c++)
                            flat[ch*49 + r*7 + c] = pool2_out[ch][r][c];

                // FC1 + ReLU
                vector<double> fc1_out;
                fc_forward(flat, W_fc1_, b_fc1_, fc1_out);
                for (double& v : fc1_out) if (v < 0) v = 0; // ReLU

                // FC2 + Softmax
                vector<double> fc2_out;
                fc_forward(fc1_out, W_fc2_, b_fc2_, fc2_out);
                softmax(fc2_out);

                // ==== 反向传播 ====
                // 输出层梯度: dL/dz = softmax - y_true
                vector<double> grad_fc2(10);
                for (int k = 0; k < 10; k++)
                    grad_fc2[k] = fc2_out[k] - (y[id] == k ? 1.0 : 0.0);

                // FC2 backward
                vector<double> grad_fc1(128, 0);
                vector<vector<double>> tmpW2(128, vector<double>(10, 0));
                vector<double> tmpb2(10, 0);
                fc_backward(fc1_out, grad_fc2, W_fc2_, grad_fc1, tmpW2, tmpb2);
                for (int i2 = 0; i2 < 128; i2++)
                    for (int o = 0; o < 10; o++)
                        dW_fc2[i2][o] += tmpW2[i2][o];
                for (int o = 0; o < 10; o++) db_fc2[o] += tmpb2[o];

                // FC1 ReLU backward
                for (int j = 0; j < 128; j++)
                    if (fc1_out[j] <= 0) grad_fc1[j] = 0;

                // FC1 backward
                vector<double> grad_flat(784, 0);
                vector<vector<double>> tmpW1(784, vector<double>(128, 0));
                vector<double> tmpb1(128, 0);
                fc_backward(flat, grad_fc1, W_fc1_, grad_flat, tmpW1, tmpb1);
                for (int i2 = 0; i2 < 784; i2++)
                    for (int o = 0; o < 128; o++)
                        dW_fc1[i2][o] += tmpW1[i2][o];
                for (int o = 0; o < 128; o++) db_fc1[o] += tmpb1[o];

                // Unflatten: 784 → [16][7][7]
                vector<vector<vector<double>>> grad_pool2(16, vector<vector<double>>(7, vector<double>(7)));
                for (int ch = 0; ch < 16; ch++)
                    for (int r = 0; r < 7; r++)
                        for (int c = 0; c < 7; c++)
                            grad_pool2[ch][r][c] = grad_flat[ch*49 + r*7 + c];

                // Pool2 backward
                vector<vector<vector<double>>> grad_conv2(16, vector<vector<double>>(14, vector<double>(14, 0)));
                maxpool_backward(grad_pool2, max_idx2, grad_conv2);

                // Conv2 ReLU backward
                relu_backward(conv2_out, grad_conv2);

                // Conv2 backward
                vector<vector<vector<double>>> grad_pool1;
                auto dW2_tmp = W_conv2_;
                for (auto& a : dW2_tmp) for (auto& b2 : a) for (auto& c2 : b2) fill(c2.begin(), c2.end(), 0.0);
                vector<double> db2_tmp(16, 0);
                conv2d_backward(pool1_out, grad_conv2, W_conv2_, grad_pool1, dW2_tmp, db2_tmp);
                for (int co = 0; co < 16; co++) {
                    db2[co] += db2_tmp[co];
                    for (int ci = 0; ci < 8; ci++)
                        for (int kr = 0; kr < 3; kr++)
                            for (int kc = 0; kc < 3; kc++)
                                dW2[co][ci][kr][kc] += dW2_tmp[co][ci][kr][kc];
                }

                // Pool1 backward
                vector<vector<vector<double>>> grad_conv1(8, vector<vector<double>>(28, vector<double>(28, 0)));
                maxpool_backward(grad_pool1, max_idx1, grad_conv1);

                // Conv1 ReLU backward
                relu_backward(conv1_out, grad_conv1);

                // Conv1 backward
                vector<vector<vector<double>>> grad_input;
                auto dW1_tmp = W_conv1_;
                for (auto& a : dW1_tmp) for (auto& b2 : a) for (auto& c2 : b2) fill(c2.begin(), c2.end(), 0.0);
                vector<double> db1_tmp(8, 0);
                conv2d_backward(img, grad_conv1, W_conv1_, grad_input, dW1_tmp, db1_tmp);
                for (int co = 0; co < 8; co++) {
                    db1[co] += db1_tmp[co];
                    for (int ci = 0; ci < 1; ci++)
                        for (int kr = 0; kr < 3; kr++)
                            for (int kc = 0; kc < 3; kc++)
                                dW1[co][ci][kr][kc] += dW1_tmp[co][ci][kr][kc];
                }
            }

            // ==== 更新参数 ====
            double inv = 1.0 / batch_n;

            for (int co = 0; co < 8; co++) {
                b_conv1_[co] -= lr_ * db1[co] * inv;
                for (int ci = 0; ci < 1; ci++)
                    for (int kr = 0; kr < 3; kr++)
                        for (int kc = 0; kc < 3; kc++)
                            W_conv1_[co][ci][kr][kc] -= lr_ * dW1[co][ci][kr][kc] * inv;
            }

            for (int co = 0; co < 16; co++) {
                b_conv2_[co] -= lr_ * db2[co] * inv;
                for (int ci = 0; ci < 8; ci++)
                    for (int kr = 0; kr < 3; kr++)
                        for (int kc = 0; kc < 3; kc++)
                            W_conv2_[co][ci][kr][kc] -= lr_ * dW2[co][ci][kr][kc] * inv;
            }

            for (int i = 0; i < 784; i++)
                for (int o = 0; o < 128; o++)
                    W_fc1_[i][o] -= lr_ * dW_fc1[i][o] * inv;
            for (int o = 0; o < 128; o++) b_fc1_[o] -= lr_ * db_fc1[o] * inv;

            for (int i = 0; i < 128; i++)
                for (int o = 0; o < 10; o++)
                    W_fc2_[i][o] -= lr_ * dW_fc2[i][o] * inv;
            for (int o = 0; o < 10; o++) b_fc2_[o] -= lr_ * db_fc2[o] * inv;
        }
        cout << "  epoch " << epoch + 1 << "/" << epochs_ << endl;
    }
}

// ====== 预测 ======

vector<int> CNN::predict(const vector<FeatureVec>& X) const {
    auto proba = predict_proba(X);
    vector<int> preds(X.size());
    for (int i = 0; i < X.size(); i++) {
        int best = 0;
        for (int k = 1; k < 10; k++)
            if (proba[i][k] > proba[i][best]) best = k;
        preds[i] = best;
    }
    return preds;
}

vector<vector<double>> CNN::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> result;
    for (const auto& x_v : X) {
        auto img = to_image(x_v);

        vector<vector<vector<double>>> conv1_out, conv1_relu;
        conv2d(img, W_conv1_, b_conv1_, conv1_out);
        conv1_relu = conv1_out;
        relu(conv1_relu);

        vector<vector<vector<double>>> pool1_out;
        vector<vector<vector<vector<int>>>> max_idx1;
        maxpool(conv1_relu, pool1_out, max_idx1);

        vector<vector<vector<double>>> conv2_out, conv2_relu;
        conv2d(pool1_out, W_conv2_, b_conv2_, conv2_out);
        conv2_relu = conv2_out;
        relu(conv2_relu);

        vector<vector<vector<double>>> pool2_out;
        vector<vector<vector<vector<int>>>> max_idx2;
        maxpool(conv2_relu, pool2_out, max_idx2);

        vector<double> flat(784);
        for (int ch = 0; ch < 16; ch++)
            for (int r = 0; r < 7; r++)
                for (int c = 0; c < 7; c++)
                    flat[ch*49 + r*7 + c] = pool2_out[ch][r][c];

        vector<double> fc1_out;
        fc_forward(flat, W_fc1_, b_fc1_, fc1_out);
        for (double& v : fc1_out) if (v < 0) v = 0;

        vector<double> fc2_out;
        fc_forward(fc1_out, W_fc2_, b_fc2_, fc2_out);
        softmax(fc2_out);
        result.push_back(fc2_out);
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
    vector<pair<double, double>> roc; roc.push_back({0.0, 0.0});
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
    EvalResult result; int N = true_labels.size();
    int correct = 0;
    for (int i = 0; i < N; i++) if (true_labels[i] == predicted[i]) correct++;
    result.accuracy = (double)correct / N;
    result.precision.resize(num_classes); result.recall.resize(num_classes);
    result.f1.resize(num_classes); result.auc.resize(num_classes);
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
        vector<int> binary_y(N); vector<double> scores(N);
        for (int i = 0; i < N; i++) { binary_y[i] = (true_labels[i] == k) ? 1 : 0; scores[i] = proba[i][k]; }
        result.roc_curves[k] = compute_roc(binary_y, scores);
        result.auc[k] = compute_auc(result.roc_curves[k]);
    }
    return result;
}

// ====== Main ======
int main() {
    cout << "Loading data..." << endl;
    auto train_data = load_images("../data/train-images-idx3-ubyte");
    auto train_labels = load_labels("../data/train-labels-idx1-ubyte");
    auto test_data = load_images("../data/t10k-images-idx3-ubyte");
    auto test_labels = load_labels("../data/t10k-labels-idx1-ubyte");
    cout << "Train: " << train_data.num_samples << " Test: " << test_data.num_samples << endl;

    int train_n = 5000, test_n = 1000;
    vector<FeatureVec> train_X(train_data.images.begin(), train_data.images.begin() + train_n);
    vector<int> train_y(train_labels.begin(), train_labels.begin() + train_n);
    vector<FeatureVec> test_X(test_data.images.begin(), test_data.images.begin() + test_n);
    vector<int> test_y(test_labels.begin(), test_labels.begin() + test_n);

    CNN cnn(0.01, 32, 10);
    cnn.fit(train_X, train_y);

    cout << "\nPredicting..." << endl;
    auto preds = cnn.predict(test_X);
    auto proba = cnn.predict_proba(test_X);
    auto result = evaluate(test_y, preds, proba, 10);

    cout << "\nAccuracy: " << result.accuracy * 100 << "%" << endl;
    cout << "\nDigit Precision Recall F1      AUC" << endl;
    cout << "----------------------------------------" << endl;
    for (int k = 0; k < 10; k++) {
        cout << " " << k << "    " << result.precision[k] << "     "
             << result.recall[k] << "     " << result.f1[k] << "     "
             << result.auc[k] << endl;
    }
    return 0;
}
