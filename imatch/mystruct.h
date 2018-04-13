#pragma once
#include "stdafx.h"
using namespace std;
using namespace cv;
//HOPC匹配点对结构体
struct match_point {
	Point2f left;//左像点
	Point2f right;//右像点
	double measure;//相似性度量值
	bool operator<(const match_point& m)
	{
		return measure < m.measure;
	}
};