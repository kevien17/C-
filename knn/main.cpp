#include "knn.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include "mnist.h"
#include "metrics.h"
#include "viz.h"
using namespace std;



//————————————————————————————————————————————————数据加载部分——————————————————————————————————————————————————————————————


// 从文件里读 4 个字节，转成大端序整数
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
















//————————————————————————————————————————————————评测部分——————————————————————————————————————————————————————————————

//Accuracy：全对的占比 = (80+890)/1000 = 97%
//Precision（查准）：预测成"3"的里面有几个真对了 = 80/(80+10) = 88.9%
//Recall（查全）：真正的"3"被你找出几个 = 80/(80+20) = 80%
//F1：Precision 和 Recall 的调和平均 = 2×0.889×0.8/(0.889+0.8) = 84.2%
//AUC：画 ROC 曲线下的面积，越大越好，最大 1.0
//ROC 曲线：横轴"宁可错杀一千"（FPR），纵轴"绝不放过一个"（TPR）

// ---------- 算 ROC 曲线 ----------
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





//————————————————————————————————————————————————可视化部分——————————————————————————————————————————————————————————————



PPMImage::PPMImage(int width, int height)
    : w_(width), h_(height), pixels_(width * height * 3, 255) {}

void PPMImage::set_pixel(int x, int y, int r, int g, int b) {
    if (x < 0 || x >= w_ || y < 0 || y >= h_) return;
    int idx = (y * w_ + x) * 3;
    pixels_[idx]     = r;
    pixels_[idx + 1] = g;
    pixels_[idx + 2] = b;
}

void PPMImage::draw_line(int x1, int y1, int x2, int y2, int r, int g, int b) {
    // Bresenham 算法——画直线最经典的方法，不用浮点数
    int dx = abs(x2 - x1);
    int dy = -abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx + dy;

    while (true) {
        set_pixel(x1, y1, r, g, b);
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x1 += sx; }
        if (e2 <= dx) { err += dx; y1 += sy; }
    }
}

void PPMImage::fill_rect(int x, int y, int w, int h, int r, int g, int b) {
    int x0 = max(0, x), y0 = max(0, y);
    int x1 = min(w_, x + w), y1 = min(h_, y + h);
    for (int iy = y0; iy < y1; iy++)
        for (int ix = x0; ix < x1; ix++)
            set_pixel(ix, iy, r, g, b);
}

void PPMImage::draw_rect(int x, int y, int w, int h, int r, int g, int b) {
    draw_line(x, y, x + w - 1, y, r, g, b);
    draw_line(x + w - 1, y, x + w - 1, y + h - 1, r, g, b);
    draw_line(x + w - 1, y + h - 1, x, y + h - 1, r, g, b);
    draw_line(x, y + h - 1, x, y, r, g, b);
}

void PPMImage::draw_digit(const vector<double>& img, int dst_x, int dst_y) {
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            int gray = (int)((1.0 - img[row * 28 + col]) * 255);
            set_pixel(dst_x + col, dst_y + row, gray, gray, gray);
        }
    }
}

bool PPMImage::save(const string& filepath) {
    ofstream f(filepath, ios::binary);
    if (!f) return false;
    f << "P6\n" << w_ << " " << h_ << "\n255\n";
    f.write((const char*)pixels_.data(), pixels_.size());
    cout << "保存: " << filepath << endl;
    return true;
}
// ---------- 画 ROC 曲线 ----------
void draw_roc_curves(const EvalResult& result, const string& outpath) {
    int W = 500, H = 500;
    int plot_x = 60, plot_y = 30;
    int plot_w = 400, plot_h = 400;

    PPMImage img(W, H);
    img.fill_rect(0, 0, W, H, 255, 255, 255);  // 白色背景

    // 坐标轴
    img.draw_line(plot_x, plot_y, plot_x, plot_y + plot_h, 0, 0, 0);
    img.draw_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, 0, 0, 0);

    // 对角虚线（随机猜测参考线）
    for (double d = 0; d < 1.0; d += 0.02) {
        int x1 = plot_x + (int)(d * plot_w);
        int y1 = plot_y + plot_h - (int)(d * plot_h);
        int x2 = plot_x + (int)(min(d + 0.01, 1.0) * plot_w);
        int y2 = plot_y + plot_h - (int)(min(d + 0.01, 1.0) * plot_h);
        img.draw_line(x1, y1, x2, y2, 200, 200, 200);
    }

    // 10 条 ROC 曲线，用不同灰度
    for (int k = 0; k < 10; k++) {
        int gray = 25 * k;  // 0, 25, 50, ..., 225
        const auto& curve = result.roc_curves[k];

        for (int i = 1; i < curve.size(); i++) {
            int x1 = plot_x + (int)(curve[i - 1].first * plot_w);
            int y1 = plot_y + plot_h - (int)(curve[i - 1].second * plot_h);
            int x2 = plot_x + (int)(curve[i].first * plot_w);
            int y2 = plot_y + plot_h - (int)(curve[i].second * plot_h);
            img.draw_line(x1, y1, x2, y2, gray, gray, gray);
        }
    }

    img.save(outpath);
}











//————————————————————————————————————————————————knn部分——————————————————————————————————————————————————————————————


KNN::KNN(int k) : k(k) {}

void KNN::fit(const vector<FeatureVec>& x, const vector<int>& y) {
    train_x = x;
    train_y = y;
}

//欧式距离计算函数/*  */
double KNN::distance(const FeatureVec& a, const FeatureVec& b) const {
    double sum = 0.0;
    for (int i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}



int KNN::vote(const vector<pair<double, int>>& neighbors) const {
    int votes[10] = {0};                    // 10 个数字类别
    for (int i = 0; i < k; i++) {         // 取前 k 个邻居
        int label = neighbors[i].second;
        votes[label]++;
    }
    int best = 0;
    for (int i = 1; i < 10; i++) {         // 找 10 个类别(数字有零到九，10个类别)里票最多的
        if (votes[i] > votes[best]) {
            best = i;
        }
  }
    return best;
}

//概率计算函数
vector<vector<double>> KNN::predict_proba(const vector<FeatureVec>& X) const {
    vector<vector<double>> result;

    for (int i = 0; i < X.size(); i++) {
        // 算距离、排序——跟 predict 一样
        vector<pair<double, int>> neighbors;
        for (int j = 0; j < train_x.size(); j++) {
            double d = distance(X[i], train_x[j]);
            neighbors.push_back({d, train_y[j]});
        }
        sort(neighbors.begin(), neighbors.end());

        // 统计前 K 个邻居里每个类别的比例
        int votes[10] = {0};
        for (int j = 0; j < k; j++) {
            votes[neighbors[j].second]++;
        }

        // 把票数转成概率（除以 K）
        vector<double> proba(10);
        for (int j = 0; j < 10; j++) {
            proba[j] = (double)votes[j] / k;
        }

        result.push_back(proba);
    }

    return result;
}

vector<int> KNN::predict(const vector<FeatureVec>& X) const {
    vector<int> result;

    // 外层循环：遍历每一张测试图片
    for (int i = 0; i < X.size(); i++) {
        // 1. 算当前测试图片到所有训练图片的距离
        vector<pair<double, int>> neighbors;
        for (int j = 0; j < train_x.size(); j++) {
            double d = distance(X[i], train_x[j]);
            neighbors.push_back({d, train_y[j]});
        }

        // 2. 按距离从小到大排序
        sort(neighbors.begin(), neighbors.end());

        // 3. 投票
        int pred = vote(neighbors);

        // 4. 存结果
        result.push_back(pred);
    }

    return result;
}













//————————————————————————————————————————————————主函数部分——————————————————————————————————————————————————————————————

int main() {
    // 1. 加载数据（路径从 knn/ 往上一级到 data/）
    cout << "加载训练图片..." << endl;
    auto train_data = load_images("../data/train-images-idx3-ubyte");
    auto train_labels = load_labels("../data/train-labels-idx1-ubyte");

    cout << "加载测试图片..." << endl;
    auto test_data = load_images("../data/t10k-images-idx3-ubyte");
    auto test_labels = load_labels("../data/t10k-labels-idx1-ubyte");

    cout << "训练集: " << train_data.num_samples << " 张"
         << " (" << train_data.rows << "x" << train_data.cols << ")" << endl;
    cout << "测试集: " << test_data.num_samples << " 张" << endl;

    // 2. 先只用 1000 张训练、200 张测试跑（全量太他妈慢了）
    int train_n = 1000;
    int test_n = 200;

    vector<FeatureVec> train_X(train_data.images.begin(),
                               train_data.images.begin() + train_n);
    vector<int> train_y(train_labels.begin(),
                        train_labels.begin() + train_n);

    vector<FeatureVec> test_X(test_data.images.begin(),
                              test_data.images.begin() + test_n);
    vector<int> test_y(test_labels.begin(),
                       test_labels.begin() + test_n);

    cout << "实际使用: 训练 " << train_n << " 张, 测试 " << test_n << " 张" << endl;

    // 3. 训练 + 预测
    KNN knn(5);  // K=5
    knn.fit(train_X, train_y);

    cout << "开始预测..." << endl;
    auto preds = knn.predict(test_X);

        // 4. 完整评测
    auto proba = knn.predict_proba(test_X);
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
        // 5. 画样本展示图
    const int COLS = 10, ROWS = 5;
    int img_w = COLS * 28 + 11 * 2;
    int img_h = ROWS * 28 + 6 * 2;
    PPMImage sample_grid(img_w, img_h);

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            int idx = r * COLS + c;
            if (idx < train_data.num_samples) {
                int dx = 2 + c * 30;
                int dy = 2 + r * 30;
                sample_grid.draw_digit(train_data.images[idx], dx, dy);
            }
        }
    }
    sample_grid.save("output/sample_grid.ppm");
        draw_roc_curves(result, "output/roc_curves.ppm");
    return 0;
}



