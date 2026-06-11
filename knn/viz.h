#pragma once
#include <vector>
#include <string>
#include "types.h"
#include "metrics.h"
using namespace std;

class PPMImage {
public:
    PPMImage(int width, int height);

    // 基础操作
    void set_pixel(int x, int y, int r, int g, int b);
    void draw_line(int x1, int y1, int x2, int y2, int r, int g, int b);
    void fill_rect(int x, int y, int w, int h, int r, int g, int b);
    void draw_rect(int x, int y, int w, int h, int r, int g, int b);

    // 画 MNIST 图片
    void draw_digit(const vector<double>& img, int dst_x, int dst_y);

    bool save(const string& filepath);

private:
    int w_, h_;
    vector<unsigned char> pixels_;  // 按 RGBRGB... 存放
};
void draw_roc_curves(const EvalResult& result, const string& outpath);