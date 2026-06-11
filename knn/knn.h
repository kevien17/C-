#pragma once
#include <vector>
#include "types.h"
using namespace std;



class KNN {
public:
    KNN(int k = 5);
    void fit(const vector<FeatureVec>& x, const vector<int>& y);
    vector<int> predict(const vector<FeatureVec>& x) const;
    vector<vector<double>> predict_proba(const vector<FeatureVec>& X) const;

private:
    int k;
    vector<FeatureVec> train_x;//二维数组，第一个数是第某张图，第二个数是该图的某个像素点的值
    vector<int> train_y;

    double distance(const FeatureVec& a, const FeatureVec& b) const;
    int vote(const vector<pair<double, int>>& neighbors) const;
};