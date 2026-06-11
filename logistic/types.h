#pragma once
#include <vector>
using namespace std;

// 一张图片就是 784 个浮点数
using FeatureVec = vector<double>;

// 数据集：一摞图片 + 对应的标签
struct Dataset {
    vector<FeatureVec> images;  // 所有图片
    vector<int> labels;         // 每个图片对应的数字（0~9）
    int num_samples = 0;        // 总共几张
    int rows = 28;              // 图片高度
    int cols = 28;              // 图片宽度
};