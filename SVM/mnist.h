#pragma once
#include "types.h"
#include <string>

// 读取图片文件，返回 Dataset
Dataset load_images(const string& filepath);

// 读取标签文件，返回 vector<int>
vector<int> load_labels(const string& filepath);