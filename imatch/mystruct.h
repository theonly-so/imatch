#pragma once
#include "stdafx.h"
using namespace std;
using namespace cv;
//HOPCƥ���Խṹ��
struct match_point {
	Point2f left;//�����
	Point2f right;//�����
	double measure;//�����Զ���ֵ
	bool operator<(const match_point& m)
	{
		return measure < m.measure;
	}
};