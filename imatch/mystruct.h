#pragma once
#include "stdafx.h"
using namespace std;
using namespace cv;
//HOPCƥ�����ṹ��
struct matches_HOPC {
	vector<Point2f> left; //�ο�Ӱ������
	vector<Point2f> right;//����Ӱ������
	double ratio;//ƥ������ȷ��
};