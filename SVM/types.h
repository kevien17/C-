#pragma once
#include <vector>
using namespace std;

using FeatureVec = vector<double>;

struct Dataset {
    vector<FeatureVec> images;
    vector<int> labels;
    int num_samples = 0;
    int rows = 28;
    int cols = 28;
};