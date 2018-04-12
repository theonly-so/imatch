// imatch.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mystruct.h"
#include <fstream>
//#include<fftw3.h>
using namespace cv;
matches_HOPC HOPC_match(Mat im_Ref, Mat im_Sen, const char* CP_Check_file, double disthre = 1.5, int tranFlag = 3, int templatesize = 100, int searchRad = 10);
Mat HarrisValue(Mat &inputimg);
vector<Point> nonmaxsupptsgrid(Mat& cim, int radius, double thresh, int gridNum, int PTnum);
vector<Mat> phasecong_hopc(Mat &im, int nscale, int norient);
Mat denseBlockHOPC(vector<Mat> &PC, const int blocksize = 3, const int cellsize = 4, const int oribins = 8);
Mat getDesc(const Mat& descriptors, const int row, const int col, const int interval, const int template_radius);
bool descNormalize(Mat& desc, int type);
double calculateCoe(const Mat& X, const Mat& Y);

int main()
{
	Mat im_Ref = imread("..\\data\\optical_ref.png");
	Mat im_Sen = imread("..\\data\\SAR_sen.png");
	const char* checkfile = "..\\data\\OpticaltoSAR_CP.txt";
	HOPC_match(im_Ref, im_Sen, checkfile);
    return 0;
}

matches_HOPC HOPC_match(Mat im_Ref,Mat im_Sen,const char* CP_Check_file,double disthre,int tranFlag,int templatesize,int searchRad)
{
	//输入参数 :
	//			im_Ref : 参考影像
	//			im_Sen : 搜索影像
	//			CP_Check_file : 检查点文件，用来确定搜索区域和判断匹配结果是否正确
	//			disthre : 匹配结果正确性判定阈值，小于该值认为结果正确
	//			tranFlag : 两张影像间的几何变换方式 0 : 仿射, 1 : 透视, 2 : 二次多项式, 3 : 三次多项式, 默认为3
	//			templateSize : 模版大小, 必须大于等于 20, 默认 100
	//			searchRad : 搜索半径, 默认10。 应小于 20

	//转变为灰度图像
	if (im_Ref.channels() == 3)
	{
		cvtColor(im_Ref, im_Ref, CV_BGR2GRAY);
	}
	if (im_Sen.channels() == 3)
	{
		cvtColor(im_Sen, im_Sen, CV_BGR2GRAY);
	}
	//转换成double 
	Mat tmp(Size(im_Ref.cols,im_Ref.rows), CV_64FC1);
	im_Ref.convertTo(tmp, CV_64FC1);
	im_Ref = tmp.clone();
	tmp.create(Size(im_Sen.cols, im_Sen.rows), CV_64FC1);
	im_Sen.convertTo(tmp, CV_64FC1);
	im_Sen = tmp.clone();
	tmp.release();


	//定义一些必须的参数
	int templateRad = cvRound(templatesize / 2);//模版半径
	int marg = templateRad + searchRad + 2;//不处理的边界宽度

	size_t C = 0;//正确匹配数
	size_t CM = 0; //总匹配数
	size_t C_e = 0; //错误匹配数
	double e = 0.0000001; // 小分母  避免除数为0

	Mat cim = HarrisValue(Mat(im_Ref, Rect(marg - 1, marg - 1, im_Ref.cols - 2 * marg + 1, im_Ref.rows - 2 * marg + 1)));
	
	vector<Point> xy=nonmaxsupptsgrid(cim, 3, 0.3, 5, 8);
	//xy变换到原图像上的位置
	for (vector<Point>::iterator i = xy.begin(); i != xy.end(); i++)
	{
		i->x += (marg - 1);
		i->y += (marg - 1);
	}
	vector<Mat> RefPC, SenPC;
	int interval = 6;
	RefPC = phasecong_hopc(im_Ref, 3, 6);
	SenPC = phasecong_hopc(im_Sen, 3, 6);
	//计算dense discriptor
	Mat denseRef = denseBlockHOPC(RefPC);
	Mat denseSen = denseBlockHOPC(SenPC);
	//计算两张影像的仿射变换参数
	vector<Point2f> refpt, senpt;
	ifstream fin(CP_Check_file);
	while (!fin.eof())
	{
		Point2f pt1, pt2;
		fin >> pt1.x >> pt1.y >> pt2.x >> pt2.y;
		refpt.push_back(pt1);
		senpt.push_back(pt2);
	}
	//计算两张影像的仿射变换参数
	fin.close();
	for (int n = 0; n < xy.size(); n++)
	{
		//判断是否在边界内

		Mat des1 = getDesc(denseRef, int(xy[n].y), int(xy[n].x), interval,templateRad);
		//归一化
		descNormalize(des1, NORM_L2);
		int si, sj;//搜索中心 现在是和左图一样
		si = (int)xy[n].y;
		sj = (int)xy[n].x;
		if (si < marg || sj < marg || si >= im_Sen.rows - marg || sj >= im_Sen.cols - marg)
			continue;
		//计算相关系数
		vector<double> matchvalue((2*searchRad+1)*(2*searchRad+1),-1.);
		for (int i = -searchRad; i <= searchRad; i++)
		{
			for (int j = -searchRad; j <= searchRad; j++)
			{
				int r = si + i;
				int c = sj + j;
				Mat des2=getDesc(denseSen, r, c, interval, templateRad);
				descNormalize(des2, NORM_L2);
				matchvalue[(i+searchRad)*(2 * searchRad + 1)+j+searchRad]=calculateCoe(des1, des2);
			}
		}
		//寻找相关系数最大的点
		vector<double>::iterator maxvalue = max_element(matchvalue.begin(), matchvalue.end());
		int maxindex = distance(matchvalue.begin(), maxvalue);

	}
	return{};
}
Mat Gaussian_kernal(int kernel_size, double sigma)
{
	const double PI = 3.14159265358979323846;
	int m = kernel_size / 2;
	Mat kernel(kernel_size, kernel_size, CV_64FC1);
	double s = 2 * sigma*sigma;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			double x = i - m, y = j - m;
			if (kernel_size % 2 == 0) {
				x += 0.5;
				y += 0.5;
			}
			kernel.ptr<double>(i)[j] = exp(-(x*x + y*y) / s) / (s*PI);
		}
	}
	return kernel;
}

Mat HarrisValue(Mat &inputimg)
{
	double sigma = 1.5;
	//注意inputimg是不是double类型
	double s_D = 0.7*sigma;
	int min = -cvRound(3 * s_D);
	//创建卷积模版 dx dy
	Mat x(1, -min * 2 + 1, CV_64FC1);
	for (int i = min; i <= -min; i++)
	{
		x.at<double>(0, i - min) = i;
	}
	Mat dx = (-1 * x).mul(x)/(2*s_D*s_D);
	for (int i = 0; i < -min*2+1; i++)
	{
		dx.at<double>(0, i) = exp(dx.at<double>(0, i));
	}
	dx = x.mul(dx) / (s_D*s_D*s_D*sqrt(2 * CV_PI));
	Mat dy = dx.t();

	//卷积操作  符号相反 边界补零计算
	Mat Ix,Iy;
	filter2D(inputimg, Ix, inputimg.depth(), dx);
	filter2D(inputimg, Iy, inputimg.depth(), dy);

	//自相关矩阵求和
	double s_I = sigma;
	Mat g = Gaussian_kernal(max(1, int(6 * s_I + 1)), s_I);
	Mat Ix2, Iy2, Ixy;
	Point cen(int((int(6 * s_I + 1) - 0.5) / 2), int((int(6 * s_I + 1) - 0.5) / 2));
	filter2D(Ix.mul(Ix), Ix2, Ix.depth(), g, cen);
	filter2D(Iy.mul(Iy), Iy2, Iy.depth(), g, cen);
	filter2D(Ix.mul(Iy), Ixy, Ix.depth(), g, cen);
	
	Mat cim = (Ix2.mul(Iy2) - Ixy.mul(Ixy));
	cvDiv(&IplImage(cim), &IplImage(Ix2 + Iy2 + 2.2204e-16), &IplImage(cim));
	
	return cim;
}

void insert_sort(double *arr,int len, string order = "ascend")
{
	//升序
	if (order == "ascend")
	{
		for (int t = 1; t < len; t++)
		{
			double key = arr[t];
			int k = t - 1;
			while (k >= 0 && arr[k] > key)
			{
				arr[k + 1] = arr[k];
				k--;
			}
			arr[k + 1] = key;
		}
	}
	//降序
	if (order == "descend")
	{
		for (int t = 1; t < len; t++)
		{
			double key = arr[t];
			int k = t - 1;
			while (k >= 0 && arr[k] < key)
			{
				arr[k + 1] = arr[k];
				k--;
			}
			arr[k + 1] = key;
		}
	}
	
}

Mat ordfilt2D(const Mat& inputimg, const int order, int size)
{
	//顺序统计量滤波，窗口大小size*size,升序排列取第order个,size必须为奇数
	Mat output = inputimg.clone();
	double *tmp = new double[size*size];
	for(int i=0;i<inputimg.rows;i++)
		for (int j = 0; j < inputimg.cols; j++)
		{
			int hw = size / 2;
			for(int m=-hw;m<=hw;m++)
				for (int n = -hw; n <= hw; n++)
				{
					if (i + m < 0 || i + m >= inputimg.rows || j + n < 0 || j + n >= inputimg.cols)
					{
						tmp[(m + hw)*size + n + hw] = 0;
					}
					else {
						tmp[(m + hw)*size + n + hw] = inputimg.ptr<double>(i + m)[j + n];
					}
				}
			//插入排序
			insert_sort(tmp, size*size);
			output.ptr<double>(i)[j] = tmp[order - 1];
		}
	delete[] tmp;
	return output;
}

vector<Point> nonmaxsupptsgrid(Mat& cim,int radius,double thresh,int gridNum,int PTnum) 
{
	//输入参数 :
	//			cim : 角点强度图像
	//			radius : 非极大值抑制的半径 一般为1-3个像素
	//			thresh : 阈值
	//			gridNum : 格网数
	//			PTnum : 每个格网的特征点个数
	//返回值：
	//			特征点的行列号
	vector<Point> xy;
	int sze = 2 * radius + 1;
	Mat mx = ordfilt2D(cim, sze*sze, sze);
	Mat bodermask=Mat::zeros(mx.rows, mx.cols, CV_8U);

	for(int i=radius;i<bodermask.rows-radius;i++)
		for (int j = radius; j < bodermask.cols - radius; j++)
		{
			bodermask.at<uchar>(i, j) = 255;
		}
	Mat tmp = (cim == mx)&(cim > thresh)&bodermask;
	tmp = tmp / 255;
	Mat cimmx(mx.rows, mx.cols, CV_64F);
	tmp.convertTo(cimmx, CV_64F);
	cimmx = cimmx.mul(cim);
	//每个格网取前8个点
	int rinterval = cim.rows / gridNum;
	int cinterval = cim.cols / gridNum;
	double *gridTmp = new double[rinterval*cinterval];
	for (int i = 0; i < gridNum; i++)
	{
		for (int j = 0; j < gridNum; j++)
		{
			for (int m = 0; m < rinterval; m++)
			{
				for (int n = 0; n < cinterval; n++)
				{
					gridTmp[m*cinterval + n] = cimmx.at<double>(m + i*rinterval, n + j*cinterval);
					//printf("%lf ", cimmx.at<double>(m + i*rinterval, n + j*cinterval));
				}
				//printf("\n");
			}
			insert_sort(gridTmp, rinterval*cinterval, "descend");
			double boundary = gridTmp[PTnum];
			for (int m = 0; m < rinterval; m++)
			{
				for (int n = 0; n < cinterval; n++)
				{
					if (cimmx.at<double>(m + i*rinterval, n + j*cinterval) > boundary)
					{
						xy.push_back(Point(n + j*cinterval, m + i*rinterval));
					}
				}
			}
		}
	}
	delete[] gridTmp;
	return xy;
}

Mat absComplex(Mat &input)
{
	//求复数的模
	Mat out(input.rows,input.cols, CV_64F);
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			out.at<double>(i, j) = sqrt(input.at<double>(i, j * 2)*input.at<double>(i, j * 2) + input.at<double>(i, j * 2 + 1)*input.at<double>(i, j * 2 + 1));
		}
	}
	return out;
}

vector<Mat> phasecong_hopc(Mat &im, int nscale, int norient)
{
	//输入参数 :
	//			im : 待处理的灰度图像
	//			nscale : log gabor尺度的数量
	//			norient : log gabor方向的数量
	//返回值：
	//			特征点的行列号

	//预定义变量
	double dThetaOnSigma = 1.7;
	double minWaveLength = 3;
	double  sigmaOnf = 0.55;
	double thetaSigma = CV_PI / norient / dThetaOnSigma;
	double mult = 2.0;
	double epsilon = 0.0001;//避免除数为0
	double g = 10;
	double k = 3;
	double cutOff = 0.4;

	
	int rows = im.rows;
	int cols = im.cols;

	vector<double> estMeanE2n;
	Mat totalSumAn = Mat::zeros(rows, cols, CV_64F);
	Mat totalEnergy = Mat::zeros(rows, cols, CV_64F);
	Mat covx2 = Mat::zeros(rows, cols, CV_64F);
	Mat covy2 = Mat::zeros(rows, cols, CV_64F);
	Mat covxy = Mat::zeros(rows, cols, CV_64F);
	Mat covx22 = Mat::zeros(rows, cols, CV_64F);
	Mat covy22 = Mat::zeros(rows, cols, CV_64F);
	Mat EnergyV2 = Mat::zeros(rows, cols, CV_64F);
	Mat EnergyV3 = Mat::zeros(rows, cols, CV_64F);
	//opencv傅立叶变换
	Mat mergelist[] ={im.clone() ,Mat::zeros(im.size(),CV_64F) };
	Mat imagefft;
	merge(mergelist, 2, imagefft);
	dft(imagefft, imagefft);

	Mat tmp1 = Mat::ones(Size(1, rows), CV_64FC1);
	Mat tmp2(Size(cols, 1), CV_64FC1);
	for (int j = 0; j < cols; j++)
	{
		tmp2.at<double>(0, j) = (j - double(cols) / 2)/(double(cols) / 2);
	}
	Mat x = tmp1*tmp2;
	tmp1.release();
	tmp2.release();
	tmp1.create(Size(1, rows),CV_64FC1);
	for (int i = 0; i < rows; i++)
	{
		tmp1.at<double>(i, 0) = (i - double(rows) / 2) / (double(rows) / 2);
	}
	tmp2 = Mat::ones(Size(cols,1),CV_64FC1);
	Mat y = tmp1*tmp2;
	Mat radius = x.mul(x) + y.mul(y);
	for (int i = 0; i < radius.rows; i++)
	{
		for (int j = 0; j < radius.cols; j++)
		{
			radius.at<double>(i, j) = sqrt(radius.at<double>(i, j));
		}
	}
	radius.at<double>(rows / 2, cols / 2) = 1;
	Mat theta = Mat::zeros(rows, cols, CV_64FC1);
	Mat sintheta = Mat::zeros(rows, cols, CV_64FC1);
	Mat costheta = Mat::zeros(rows, cols, CV_64FC1);

	for (int i = 0; i < theta.rows; i++)
	{
		for (int j = 0; j < theta.cols; j++)
		{
			theta.at<double>(i, j) = atan2(-y.at<double>(i,j), x.at<double>(i, j));
			sintheta.at<double>(i, j) = sin(theta.at<double>(i, j));
			costheta.at<double>(i, j) = cos(theta.at<double>(i, j));
		}
	}
	for (int o = 1; o <= norient; o++)
	{
		printf("处理角度%d\n", o);
		double angl= double(o - 1) * CV_PI / norient;
		double wavelength = minWaveLength;
		Mat ds = sintheta*cos(angl) - costheta*sin(angl);
		Mat dc = costheta*cos(angl) + sintheta*sin(angl);
		Mat spread= Mat::zeros(rows, cols, CV_64FC1);
		Mat sumE_ThisOrient = Mat::zeros(rows, cols, CV_64F);
		Mat	sumO_ThisOrient = Mat::zeros(rows, cols, CV_64F);
		Mat sumO_ThisOrient1 = Mat::zeros(rows, cols, CV_64F);
		Mat	sumAn_ThisOrient = Mat::zeros(rows, cols, CV_64F);
		Mat Energy_ThisOrient = Mat::zeros(rows, cols, CV_64F);
		Mat maxSum0, maxAn;
		double EM_n;
		for (int i = 0; i < spread.rows; i++)
		{
			for (int j = 0; j < spread.cols; j++)
			{
				spread.at<double>(i, j) = abs(atan2(ds.at<double>(i, j), dc.at<double>(i, j)));
				spread.at<double>(i, j) = exp((-spread.at<double>(i, j)*spread.at<double>(i, j)) / (2 * thetaSigma*thetaSigma));
			}
		}
		vector<Mat> EOArray, ifftFilterArray;
		for (int s = 1; s <= nscale; s++)
		{
			double fo = 1. / wavelength;
			double rfo = fo / 0.5;
			Mat logGabor = Mat::zeros(rows, cols, CV_64FC1);
			for (int i = 0; i < logGabor.rows; i++)
			{
				for (int j = 0; j < logGabor.cols; j++)
				{
					logGabor.at<double>(i, j) = log(radius.at<double>(i, j) / rfo);
					logGabor.at<double>(i, j) = -logGabor.at<double>(i, j)*logGabor.at<double>(i, j) / (2 * log(sigmaOnf)*log(sigmaOnf));
					logGabor.at<double>(i, j) = exp(logGabor.at<double>(i, j));
				}
			}
			logGabor.at<double>(rows / 2, cols / 2) = 0;
			logGabor = logGabor.mul(spread);
			Mat filter(rows, cols, CV_64F);
			for (int i = 0; i < logGabor.rows; i++)
			{
				for (int j = 0; j < logGabor.cols; j++)
				{
					if (i < rows / 2 && j < cols / 2)
					{
						filter.at<double>(i, j) = logGabor.at<double>(i + rows / 2, j + cols / 2);
					}
					else if (i < rows / 2 && j >= cols / 2)
					{
						filter.at<double>(i, j) = logGabor.at<double>(i + rows / 2, j - cols / 2);
					}
					else if (i >= rows / 2 && j < cols / 2)
					{
						filter.at<double>(i, j) = logGabor.at<double>(i - rows / 2, j + cols / 2);
					}
					else {
						filter.at<double>(i, j) = logGabor.at<double>(i - rows / 2, j - cols / 2);
					}
				}
			}

			//opencv傅立叶变换
			Mat mergelist2[] = { filter.clone(),Mat::zeros(filter.size(),CV_64F) };
			Mat ifftfilt;
			merge(mergelist2, 2, ifftfilt);
			idft(ifftfilt, ifftfilt);
			vector<Mat> splitlist1;
			split(ifftfilt, splitlist1);
			Mat ifftFilt = splitlist1[0].clone() / sqrt(rows*cols);
			splitlist1.clear();

			mergelist2[0] = filter.clone();
			mergelist2[1] = filter.clone();
			Mat EO;
			merge(mergelist2, 2, EO);
			EO = imagefft.mul(EO);
			idft(EO, EO);
			EO = EO / (rows*cols);

			//整合不同尺度
			ifftFilterArray.push_back(ifftFilt);
			EOArray.push_back(EO);
			//hconcat(EOArray, EO, EOArray);
			
			sumAn_ThisOrient = sumAn_ThisOrient + absComplex(EO);
			split(EO, splitlist1);
			sumE_ThisOrient = sumE_ThisOrient + splitlist1[0];
			sumO_ThisOrient = sumO_ThisOrient + splitlist1[1];
			if (s > 1)
			{
				sumO_ThisOrient1 = sumO_ThisOrient1 + splitlist1[1];
			}
			if (s == 1)
			{
				maxSum0 = sumO_ThisOrient.clone();
				maxAn = absComplex(EO);
				EM_n=sum(filter.mul(filter))[0];
			}else
			{
				maxSum0 = max(maxSum0, sumO_ThisOrient);
				maxAn = max(maxAn, absComplex(EO));
			}
			wavelength = wavelength*mult;

			//for (int i = 0; i < 20; i++)
			//{
			//	for (int j = 0; j < 20; j++)
			//	{
			//		//printf("%lf+%lfi ", EO.at<double>(i, j*2), EO.at<double>(i, j * 2+1));
			//		printf("%lf ", sumAn_ThisOrient.at<double>(i, j ));

			//	}
			//	printf("\n");
			//}
		}//处理下一个尺度
		Mat XEnergy = sumE_ThisOrient.mul(sumE_ThisOrient) + sumO_ThisOrient.mul(sumO_ThisOrient);
		for (int i = 0; i < XEnergy.rows; i++)
		{
			for (int j = 0; j < XEnergy.cols; j++)
			{
				XEnergy.at<double>(i, j) = sqrt(XEnergy.at<double>(i, j)) + epsilon;
			}
		}
		Mat MeanE, MeanO;
		divide(sumE_ThisOrient, XEnergy, MeanE);
		divide(sumO_ThisOrient, XEnergy, MeanO);
		for (int s = 1; s <= nscale; s++)
		{
			//提取奇偶滤波器
			Mat EO = EOArray[s - 1];
			vector<Mat> splitlist;
			split(EO, splitlist);
			Mat E = splitlist[0];
			Mat O = splitlist[1];
			Energy_ThisOrient = Energy_ThisOrient + E.mul(MeanE) + O.mul(MeanO) - abs(E.mul(MeanO) - O.mul(MeanE));
		}
		//求中位数
		Mat sortTmp = (absComplex(EOArray[0]).mul(absComplex(EOArray[0])));
		sort(sortTmp.ptr<double>(0), sortTmp.ptr<double>(rows-1) + cols);
		double medianE2n;
		if (rows % 2 == 0)
		{
			medianE2n = (sortTmp.at<double>(rows / 2 - 1, cols - 1) + sortTmp.at<double>(rows / 2, 0)) / 2;
		}
		else if(cols % 2 == 0)
		{
			medianE2n = (sortTmp.at<double>(rows / 2, cols / 2) + sortTmp.at<double>(rows / 2, cols / 2 - 1)) / 2;
		}else
		{
			medianE2n = sortTmp.at<double>(rows / 2, cols / 2);
		}
		double meanE2n= -medianE2n / log(0.5);
		estMeanE2n.push_back(meanE2n);
		double noisePower=meanE2n / EM_n; //估计噪声的功率
		Mat EstSumAn2 = Mat::zeros(rows, cols, CV_64F);
		for (int s = 1; s <= nscale; s++)
		{
			EstSumAn2 = EstSumAn2 + ifftFilterArray[s - 1].mul(ifftFilterArray[s - 1]);
		}
		Mat EstSumAiAj = Mat::zeros(rows, cols, CV_64F);
		for (int si = 1; si < nscale; si++)
		{
			for (int sj = si + 1; sj <= nscale; sj++)
			{
				EstSumAiAj = EstSumAiAj + ifftFilterArray[si - 1].mul(ifftFilterArray[sj - 1]);
			}
		}
		double EstNoiseEnergy2 = (2 * noisePower*sum(EstSumAn2) + 4 * noisePower*sum(EstSumAiAj))[0];
		double tau = sqrt(EstNoiseEnergy2 / 2);                    
		double EstNoiseEnergy = tau*sqrt(CV_PI / 2);                  
		double EstNoiseEnergySigma = sqrt((2 - CV_PI / 2)*(tau*tau));
		double T = EstNoiseEnergy + k*EstNoiseEnergySigma;//噪声阈值

		T = T / 1.7;
		Energy_ThisOrient = Energy_ThisOrient - T;
		Energy_ThisOrient = max(Energy_ThisOrient, Mat::zeros(rows, cols, CV_64F));
		Mat width;
		divide(sumAn_ThisOrient, (maxAn + epsilon)*nscale, width);
		Mat weight = (cutOff - width)*g;
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				weight.at<double>(i, j) = 1.0 / (1 + exp(weight.at<double>(i, j)));
			}
		}
		Energy_ThisOrient = weight.mul(Energy_ThisOrient);
		totalSumAn = totalSumAn + sumAn_ThisOrient;
		totalEnergy = totalEnergy + Energy_ThisOrient;

		Mat PC;
		divide(Energy_ThisOrient, sumAn_ThisOrient, PC);
		Mat covx = PC*cos(angl);
		Mat covy = PC*sin(angl);
		covx2 = covx2 + covx.mul(covx);
		covy2 = covy2 + covy.mul(covy);
		covxy = covxy + covx.mul(covy);
		covx22 = covx22 + covx;
		covy22 = covy22 + covy;
		EnergyV2 = EnergyV2 + cos(angl)*sumO_ThisOrient;
		EnergyV3 = EnergyV3 + sin(angl)*sumO_ThisOrient;
	}
	Mat phaseCongruency,or;
	divide(totalEnergy, totalSumAn + epsilon, phaseCongruency);
	divide(EnergyV3, -EnergyV2, or );
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			or .at<double>(i, j) = atan(or .at<double>(i, j));
		}
	}
	//for (int i = 0; i < 20; i++)
	//{
	//	for (int j = 0; j < 20; j++)
	//	{
	//		//printf("%d ", tmp.at<uchar>(i, j));
	//		//printf("%lf+%lfi ", imagefft[i*cols+j][0], imagefft[i*cols + j][1]);
	//		printf("%lf ", or.at<double>(i, j));
	//	}
	//	printf("\n");
	//}
	vector<Mat> rValue;
	rValue.push_back(phaseCongruency);
	rValue.push_back(or );
	return rValue;
}

Mat denseBlockHOPC(vector<Mat> &PC,const int blocksize, const int cellsize, const int oribins)
{
	Mat pc = PC[0];
	Mat or = PC[1];
	int rows = pc.rows;
	int cols = pc.cols;
	int dis_len = blocksize*blocksize*oribins;//每个block描述子的长度
	double angle_interval = CV_PI / oribins;
	Mat descriptors=Mat::zeros(rows, cols, CV_32FC(dis_len));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (or.at<double>(i, j) < 0)
			{
				or.at<double>(i, j) += CV_PI;
			}
		}
	}
	//计算以每个像素为中心的block区域的描述符
	
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//block
			for (int bi = 0; bi < blocksize; bi++)
			{
				for (int bj = 0; bj < blocksize; bj++)
				{
					//cell
					int thisori;
					for (int ci = 0; ci < cellsize; ci++)
					{
						for (int cj = 0; cj < cellsize; cj++)
						{
							//此像素对应的原图行列号
							int r = i + bi*blocksize + ci;
							int c = j + bj*blocksize + cj;
							if (r < 0 || c < 0 || r >= rows || c >= cols)
							{
								continue;
							}
							int n = cvFloor(or .at<double>(i, j) / angle_interval);
							//加权累加 双线性插值
							double k1 = (or .at<double>(i, j) - angle_interval*n) / angle_interval;
							double k2 = 1 - k1;
							descriptors.at<float>(i, j*dis_len + (bi*blocksize + bj)*oribins + n)+=(k2*pc.at<double>(r,c));
							if (n + 1 < oribins)
							{
								descriptors.at<float>(i, j*dis_len + (bi*blocksize + bj)*oribins + n + 1) += (k1*pc.at<double>(r, c));
							}
							else 
							{
								descriptors.at<float>(i, j*dis_len + (bi*blocksize + bj)*oribins + 0) += (k1*pc.at<double>(r, c));
							}
						}
					}
					//cell end
				}
			}
			//block end
		}
	}
	return descriptors;
}

Mat getDesc(const Mat& descriptors, const int row, const int col, const int interval,const int template_radius)
{
	//函数功能 :
	//			从整张图的密集特征返回某一个点的描述子
	//输入参数 :
	//			descriptors : 向量X
	//			row : 行
	//			col : 列
	//			interval : 间隔
	//			template_radius : 模板半径
	//返回值：
	//			9*块描述子 大小的特征向量
	int len = descriptors.channels();
	Mat result = Mat::zeros(9, len , CV_32F);
	if (row < template_radius || col < template_radius || row >= descriptors.rows - template_radius || col >= descriptors.cols - template_radius)
	{
		return result;
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int n = 0; n < len; n++)
			{
				int r = row + (i - 1)*interval;
				int c = (col + (j - 1)*interval)*len + n;
				result.at<float>(3 * i + j, n) = descriptors.at<float>(r, c);
			}
		}
	}
	return result;
}

double calculateCoe(const Mat& X,const Mat& Y)
{
	//函数功能 :
	//			计算相关系数
	//输入参数 :
	//			X : 向量X
	//			Y : 向量Y
	//返回值：
	//			相关系数
	double glgr, gl, gr, gl2, gr2;
	glgr = 0; gl = 0; gr = 0; gl2 = 0; gr2 = 0;
	int rows = X.rows;
	int cols = X.cols;
	int n_channel = X.channels();
	typedef float MATTYPE;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			for (int ch = 0; ch < n_channel; ch++)
			{
				int r = i;
				int c = j*n_channel + ch;
				glgr += X.at<float>(r,c) * Y.at<float>(r, c);
				gl += X.at<MATTYPE>(r, c);
				gr += Y.at<MATTYPE>(r, c);
				gl2 += X.at<MATTYPE>(r, c)* X.at<MATTYPE>(r, c);
				gr2 += Y.at<MATTYPE>(r, c) * Y.at<MATTYPE>(r, c);
			}		
		}
	int size = rows*cols*n_channel;
	double value = (glgr - gl*gr / size) / sqrt((gl2 - gl*gl / size)*(gr2 - gr*gr / size));
	return value;
}

bool descNormalize(Mat& desc, int type)
{
	//函数功能 :
	//			归一化特征向量(行归一化)
	//输入参数 :
	//			desc : 特征向量
	//			type : 归一化方法
	//返回值：
	//			成功返回true,失败返回false
	int rows = desc.rows;
	for (int i = 0; i < rows; i++)
	{
		normalize(desc.rowRange(i, i + 1), desc.rowRange(i, i + 1), 1, 0, type);
	}
	return true;
}