#pragma once
#include "stdafx.h"
using namespace std;
using namespace cv;
//HOPC匹配结果结构体
struct matches_HOPC {
	vector<Point2f> left; //参考影像坐标
	vector<Point2f> right;//搜索影像坐标
	double ratio;//匹配结果正确率
};