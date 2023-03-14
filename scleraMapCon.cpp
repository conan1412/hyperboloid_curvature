#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <windows.h>
#include <ShlObj.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <io.h>
#include <fstream>
#include <math.h>
#include "colormapYT.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include "scleraMapCon.h"
#include <numeric>
#include <random>
#include "omp.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include  "save_pointcloud.h"
#define _USE_MATH_DEFINES
#define M_PI 3.14159265358979323846

extern "C" void zernike_fit_cuda(int nZernike, int sizeZernike, float* h_zernikMatrix);


using namespace std;
using namespace cv;
using namespace colormapYT;

static void fill_red(Mat& mask, Mat& lines) {
	for (int i = 0; i < lines.cols; i++) {
		circle(mask, Point(lines.at<double>(0, i), lines.at<double>(1, i)), 1, Scalar(0, 0, 255), 1); //画点
	}
}
static void fill_purple(Mat& mask, Mat& lines) {
	for (int i = 0; i < lines.cols; i++) {
		circle(mask, Point(lines.at<double>(0, i), lines.at<double>(1, i)), 1, Scalar(255, 0, 255), 1); //画点
	}
}
static void fill_cyan(Mat& mask, Mat& lines) {
	for (int i = 0; i < lines.cols; i++) {
		circle(mask, Point(lines.at<double>(0, i), lines.at<double>(1, i)), 1, Scalar(255, 255, 0), 1); //画点
	}
}
static void fill_blue(Mat& mask, Mat& lines) {
	for (int i = 0; i < lines.cols; i++) {
		circle(mask, Point(lines.at<double>(0, i), lines.at<double>(1, i)), 1, Scalar(255, 0, 0), 1); //画点
	}
}
static void fill_green(Mat& mask, Mat& lines) {
	for (int i = 0; i < lines.cols; i++) {
		circle(mask, Point(lines.at<double>(0, i), lines.at<double>(1, i)), 1, Scalar(0, 255, 0), 1); //画点
	}
}
static void fill_yellow(Mat& mask, Mat& lines) {
	for (int i = 0; i < lines.cols; i++) {
		circle(mask, Point(lines.at<double>(0, i), lines.at<double>(1, i)), 1, Scalar(0, 255, 255), 1); //画点
	}
}

void save_pointcloud_aspcd(Mat cornea_map_mat, std::string filename) {

	vector<fPointXYZ> vPts;
	for (int x = 0; x < cornea_map_mat.cols;x++) {
		for (int y = 0; y < cornea_map_mat.rows;y++) {
			if (cornea_map_mat.at<double>(x, y) != 0) { // z值不为0
				fPointXYZ tp;
				tp.x = x;
				tp.y = y;
				tp.z = cornea_map_mat.at<double>(x, y);
				vPts.push_back(tp);
			}
		}
	}

	int points_len = vPts.size();
	std::ofstream fout_pc_name(filename);

	fout_pc_name << "# .PCD v0.7 - Point Cloud Data file format" << std::endl;
	fout_pc_name << "VERSION 0.7" << std::endl;
	fout_pc_name << "FIELDS x y z" << std::endl;
	fout_pc_name << "SIZE 4 4 4" << std::endl;
	fout_pc_name << "TYPE F F F" << std::endl;
	fout_pc_name << "COUNT 1 1 1" << std::endl;
	fout_pc_name << "WIDTH " << points_len << std::endl;
	fout_pc_name << "HEIGHT 1" << std::endl;
	fout_pc_name << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
	fout_pc_name << "POINTS " << points_len << std::endl;
	fout_pc_name << "DATA ascii" << std::endl;

	for (int n = 0; n < points_len; n++) {
		fout_pc_name << vPts[n].x << " " << vPts[n].y << " " << vPts[n].z << std::endl;
	}
	fout_pc_name.close();
}

Mat array2mat(double** map, int dimxy) {
	Mat map_mat = Mat::zeros(dimxy, dimxy, CV_64FC1);
	for (int x = 0; x < dimxy; ++x) {
		for (int y = 0; y < dimxy; ++y) {
			map_mat.at<double>(x, y) = map[x][y];
		}
	}
	return map_mat;
}

// 替换字符串
string& replace_all(string& src, const string& old_value, const string& new_value) {
	// 每次重新定位起始位置，防止上轮替换后的字符串形成新的old_value
	for (string::size_type pos(0); pos != string::npos; pos += new_value.length()) {
		if ((pos = src.find(old_value, pos)) != string::npos) {
			src.replace(pos, old_value.length(), new_value);
		}
		else break;
	}
	return src;
}

//比较轮廓面积(USB_Port_Lean用来进行轮廓排序)
bool Contour_Area(vector<Point> contour1, vector<Point> contour2)
{
	return contourArea(contour1) > contourArea(contour2);
}

// 替换字符串并保存图像
void replace_name_saveimg(string src, string out, Mat img) {
	string filename_tmp = src;
	filename_tmp.replace(filename_tmp.find("_"), 1, out); // "_roughRegion_"
	imwrite(filename_tmp, img);
}



// 根据半径去除离散点, 在当前点的搜索半径为radius的范围内，如果相邻点个数少于k个，则该点为离群点，进行剔除
void removeOutlier(vector<Point> inData, int radius, int k, vector<Point>& outData)
{
	outData.clear();

	int cnt = 0;
	int n = 0;
	for (int m = 0; m < inData.size(); m++)
	{
		cnt = 0;
		for (n = 0; n < inData.size(); n++)
		{
			if (n == m)
				continue;

			if (sqrt(pow(inData[m].x - inData[n].x, 2) + pow(inData[m].y - inData[n].y, 2)) <= radius)
			{
				cnt++;
				if (cnt >= k)
				{
					outData.push_back(inData[m]);
					n = 0;
					break;
				}
			}
		}
	}
}

// 使用阈值检测上下界线
vector<Point> BoundaryDetect_pixel(Mat im, int window, int cutSize)
{

	int col = im.rows; //2048
	int row = im.cols; //1049
	int col_min = window; //200
	int col_max = col - window; // 2048 -200
	//flip(im, im, 0); //上下颠倒
	//Mat CorneaFC = Mat::zeros(1, col, CV_64FC1);
	vector<Point> CorneaFC;

	//int start = 0;
	for (int i = 0; i <= row;i++) //[0,1049]
	{
		for (int j = col_min; j < col_max; j++) //[col_min:col_max]
		{
			if ((double)im.at<uint8_t>(j, i) >= 255.0)
			{
				//CorneaFC.at<double>(0, i) = (double)(j); //CorneaFC 1维度存col-j坐标，上表面坐标
				CorneaFC.push_back(Point(i + cutSize, j));
				break;
			}
		}
		//start = start + 1;
	}

	vector<Point> outData;
	//removeOutlier(CorneaFC, 50, 50, outData); //在当前点的搜索半径为radius的范围内，如果相邻点个数少于k个，则该点为离群点，进行剔除
	outData = CorneaFC;
	return outData;

	////排序，给出索引
	//double minVal, maxVal;
	//int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
	//minMaxIdx(CorneaFC, &minVal, &maxVal, minIdx, maxIdx);
	//Mat CorneaFC_tmp = CorneaFC / (maxVal - minVal);
	//vector<Point> CorneaFC_out;
	//for (int i = col_min; i <= col_max;i++) {//[0,1049]
	//	if (CorneaFC_tmp.at<double>(0, i) > 0.5  && ((i + cutSize < 1024 - 40) || (i + cutSize > 1024 + 40))) {  //取出大于0.5阈值的数
	//		CorneaFC_out.push_back(Point(i + cutSize, CorneaFC.at<double>(0, i)));  //存真实坐标
	//	}
	//}
	//return CorneaFC_out;
}

void draw_xys(Mat& display2, Mat& map, int rt, int n, double Rcornea, int addSize) {
	vector<Point> xys;
	int baseline = 0, fontScale = 1, thickness = 2;
	double init_theta = 2 * M_PI / n;
	for (int i = 0; i < n; i++) {
		Point xy = Point(int(rt * cos(init_theta * i)) + Rcornea + addSize,
			int(rt * sin(init_theta * i)) + Rcornea + addSize);
		float value = map.at<double>(xy.x, xy.y);
		stringstream ss;
		ss << "+" << fixed << setprecision(2) << value; //保留小数点后两位
		Size textSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
		putText(display2, ss.str(), Point(xy.x - textSize.width / 2, xy.y + textSize.height / 2), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness, 8);
	}

}

// 曲率图填数值
void cornea_draw_values(Mat& display2, Mat& map, double Rcornea, int addSize) {
	//画圆心
	int rt = 0, n = 1;
	draw_xys(display2, map, rt, n, Rcornea, addSize);
	rt = int(Rcornea * 5 / 12), n = 6;
	draw_xys(display2, map, rt, n, Rcornea, addSize);
	rt = int(Rcornea * 4 / 5), n = 12;
	draw_xys(display2, map, rt, n, Rcornea, addSize);
}

//---------------------------------------------------------------------------------------------------
// 实现一个类似三次样条插值，注意在我们的数据中需要做排序预处理
//---------------------------------------------------------------------------------------------------
// 样条插值的主函数
void BSpline(vector<Point2f> xy, vector<float> xx, vector<float>& yy)
{
	// xy: 点向量，欲拟合的点，类型为float
	// xx: 重新规划的x坐标
	int n = xy.size(); //18
	Mat a = Mat::zeros(n - 1, 1, CV_32FC1); //shape(17,1)

	Mat b = Mat::zeros(n - 1, 1, CV_32FC1); //shape(17,1)
	Mat d = Mat::zeros(n - 1, 1, CV_32FC1); //shape(17,1)
	Mat dx = Mat::zeros(n - 1, 1, CV_32FC1); //shape(17,1)
	Mat dy = Mat::zeros(n - 1, 1, CV_32FC1); //shape(17,1)
	for (int i = 0; i < xy.size() - 1; i++)
	{
		a.at<float>(i, 0) = xy[i].y;  //存入y
		dx.at<float>(i, 0) = (xy[i + 1].x - xy[i].x);  //存入xi+1 - xi
		dy.at<float>(i, 0) = (xy[i + 1].y - xy[i].y);  //存入yi+1 - yi
	}
	Mat A = Mat::zeros(n, n, CV_32FC1); //shape(18,18)
	Mat B = Mat::zeros(n, 1, CV_32FC1); //shape(18,1)
	A.at<float>(0, 0) = 1;
	A.at<float>(n - 1, n - 1) = 1;
	for (int i = 1; i <= n - 2; i++) //[1,17]
	{
		A.at<float>(i, i - 1) = dx.at<float>(i - 1, 0);
		A.at<float>(i, i) = 2 * (dx.at<float>(i - 1, 0) + dx.at<float>(i, 0));
		A.at<float>(i, i + 1) = dx.at<float>(i, 0);
		B.at<float>(i, 0) = 3 * (dy.at<float>(i, 0) / dx.at<float>(i, 0) - dy.at<float>(i - 1, 0) / dx.at<float>(i - 1, 0));
	}
	Mat c = A.inv() * B;
	for (int i = 0; i <= n - 2; i++)
	{
		d.at<float>(i, 0) = (c.at<float>(i + 1, 0) - c.at<float>(i, 0)) / (3 * dx.at<float>(i, 0));
		b.at<float>(i, 0) = dy.at<float>(i, 0) / dx.at<float>(i, 0) - dx.at<float>(i, 0) * (2 * c.at<float>(i, 0) + c.at<float>(i + 1, 0)) / 3;
	}
	int j;
	for (int i = 0; i < xx.size(); i++)
	{
		for (int ii = 0; ii <= n - 2; ii++)
		{
			if (xx[i] >= xy[ii].x && xx[i] < xy[ii + 1].x)
			{
				j = ii;
				break;
			}
			else if (xx[i] < xy[0].x)
			{
				j = 0;
				break;
			}
			else
			{
				j = n - 1 - 1;
			}
		}
		float middleV = a.at<float>(j, 0) + b.at<float>(j, 0) * (xx[i] - xy[j].x) + c.at<float>(j, 0) * 
			(xx[i] - xy[j].x) * (xx[i] - xy[j].x) + d.at<float>(j, 0) * (xx[i] - xy[j].x) * (xx[i] - xy[j].x) * (xx[i] - xy[j].x);
		yy.push_back(middleV);
	}
}
// 样条插值预处理函数
void BsplinePre(Mat xx, Mat zz, Mat XY_um, Mat& ZZ_c, int range) { //bspline轨迹优化
	vector<float> xx1;
	vector<float> yy1;
	vector<Point2f> xyNew;
	Point2f test;
	int number = xx.cols; //1849

	for (int i = 0; i < number;i++) {
		if ((i < number - range) && (i % range == 0)) {
			test.x = mean(xx.colRange(i, i + range)).val[0];
			test.y = mean(zz.colRange(i, i + range)).val[0];
			xyNew.push_back(test);
		}
	}
	for (int i = 0; i < XY_um.cols;i++) {
		xx1.push_back((float)XY_um.at<double>(0, i)); //xx1: XY_um[0,:]
	}

	BSpline(xyNew, xx1, yy1); //B-spline的轨迹优化,优化yy1
	for (int i = 0; i < XY_um.cols; i++) {
		ZZ_c.at<double>(0, i) = yy1[i];  //存入ZZ_c
	}
}


//---------------------------------------------------------------------------------------------------
// 利用opencv，实现一个多项式拟合
//---------------------------------------------------------------------------------------------------
//
Mat polyfit(vector<Point>& in_point, int n)
{
	int size = in_point.size();
	//所求未知数个数
	int x_num = n + 1;
	//构造矩阵U和Y
	Mat mat_u(size, x_num, CV_64F);
	Mat mat_y(size, 1, CV_64F);

	for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
			mat_u.at<double>(i, j) = pow(in_point[i].x, j);
		}

	for (int i = 0; i < mat_y.rows; ++i)
	{
		mat_y.at<double>(i, 0) = in_point[i].y;
	}

	//矩阵运算，获得系数矩阵K
	Mat mat_k(x_num, 1, CV_64F);
	mat_k = (mat_u.t() * mat_u).inv() * mat_u.t() * mat_y;
	return mat_k;
}

// 线性拟合函数
void linefit(double pointx1, double pointy1, double pointx2, double pointy2, vector<double>& vResult)
{
	if ((pointx1 - pointx2) == 0)
	{
		vResult.push_back(0.0);
		vResult.push_back(0.0);
		vResult.push_back(-1.0);
	}
	else
	{
		vResult.push_back((pointy1 - pointy2) / (pointx1 - pointx2));
		vResult.push_back(pointy1 - vResult[0] * pointx1);
		vResult.push_back(1.0);
	}
}

// 插值函数，距离反比
double pointCal(double pointx1, double pointy1, double value1, double pointx2, double pointy2, double value2, double pointx3, double pointy3, double value3, double pointx, double pointy)
{
	double distance1 = sqrt((pointx - pointx1) * (pointx - pointx1) + (pointx - pointx1) * (pointx - pointx1));
	double distance2 = sqrt((pointx - pointx2) * (pointx - pointx2) + (pointy - pointy2) * (pointy - pointy2));
	double distance3 = sqrt((pointx - pointx3) * (pointx - pointx3) + (pointy - pointy3) * (pointy - pointy3));

	double finalData;
	if (distance1 == 0)
	{
		return finalData = value1;
	}
	else if (distance2 == 0)
	{
		return finalData = value2;
	}
	else if (distance3 == 0)
	{
		return finalData = value3;
	}
	else
	{
		double alldistance = 100 / distance1 + 100 / distance2 + 100 / distance3;
		double pos1 = (100 / distance1) / alldistance;
		double pos2 = (100 / distance2) / alldistance;
		double pos3 = (100 / distance3) / alldistance;
		return finalData = pos1 * value1 + pos2 * value2 + pos3 * value3;
	}

}


// 实现类似matlab中griddata函数的功能
void twelve_line(double* mLayerx, double* mLayery, double* mThickCO, int number, int dim, double** map) {
	int half_size = floor(dim / 2);
	int x1; // 存储x坐标
	int y1; // 存储y坐标
	double z1;
	int numberData = 0; //统计需要参与剖分的数目
	// 建立剖分网络和填充已有的数据
	for (int i = 0; i < number; i++)
	{
		// 初始化坐标和高度值
		y1 = (int)round(mLayerx[i]);
		x1 = (int)round(mLayery[i]);
		// 根据matlab的结果做转置，节省计算量，直接在这里转置完毕必须是方阵
		z1 = mThickCO[i];
		// 
		if ((x1 >= -half_size) && (x1 <= half_size) && (y1 >= -half_size) && (y1 <= half_size))
		{
			x1 = x1 + half_size;
			y1 = y1 + half_size;
			map[x1][y1] = z1;
			numberData = numberData + 1;
		}
	}
}


// 实现类似matlab中griddata函数的功能
void griddataFun(double* mLayerx, double* mLayery, double* mThickCO, int number, int dim, double** map)
{
	int half_size = floor(dim / 2);
	int x1; // 存储x坐标
	int y1; // 存储y坐标
	double z1;
	int numberData = 0; //统计需要参与剖分的数目
	// 建立剖分网络和填充已有的数据
	for (int i = 0; i < number; i++)
	{
		// 初始化坐标和高度值
		y1 = (int)round(mLayerx[i]);
		x1 = (int)round(mLayery[i]);
		// 根据matlab的结果做转置，节省计算量，直接在这里转置完毕必须是方阵
		z1 = mThickCO[i];
		// 
		if ((x1 >= -half_size) && (x1 <= half_size) && (y1 >= -half_size) && (y1 <= half_size))
		{
			x1 = x1 + half_size;
			y1 = y1 + half_size;
			if (map[x1][y1] == 0)
			{
				map[x1][y1] = z1;
				numberData = numberData + 1;
			}
			else
			{
				map[x1][y1] = (map[x1][y1] + z1) / 2;
			}
		}

	}

	//figurePoint(map, dim, dim, 300.0, 800.0, "ori");

	// 初始化坐标列表
	double** sanjiaoData = new double* [numberData];
	for (int i = 0; i < numberData; i++)
	{
		sanjiaoData[i] = new double[3]();
	}
	int start = 0;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			if (map[i][j] != 0)
			{
				sanjiaoData[start][0] = (double)i;
				sanjiaoData[start][1] = (double)j;
				sanjiaoData[start][2] = map[i][j];
				start = start + 1;
			}
		}
	}
	// 三角剖分
	Rect rect(0, 0, dim, dim);
	Subdiv2D subdiv(rect);
	for (int i = 0; i < numberData; i++)
	{
		Point2f fp((float)(sanjiaoData[i][0]),
			(float)(sanjiaoData[i][1]));

		subdiv.insert(fp);
	}
	// 数据列表
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	// 空间索引和插值	
	for (int i = 0; i < triangleList.size(); i++)
	{
		double data1[3] = { (double)triangleList[i][0], (double)triangleList[i][1],map[(int)triangleList[i][0]][(int)triangleList[i][1]] };
		double data2[3] = { (double)triangleList[i][2], (double)triangleList[i][3],map[(int)triangleList[i][2]][(int)triangleList[i][3]] };
		double data3[3] = { (double)triangleList[i][4], (double)triangleList[i][5],map[(int)triangleList[i][4]][(int)triangleList[i][5]] };
		double dataPositonX[3] = { (double)triangleList[i][0], (double)triangleList[i][2], (double)triangleList[i][4] };
		double dataPositonY[3] = { (double)triangleList[i][1], (double)triangleList[i][3], (double)triangleList[i][5] };
		double dataXmax = *max_element(dataPositonX, dataPositonX + 3);
		double dataXmin = *min_element(dataPositonX, dataPositonX + 3);
		double dataYmax = *max_element(dataPositonY, dataPositonY + 3);
		double dataYmin = *min_element(dataPositonY, dataPositonY + 3);
		//
		vector<double> para1;
		linefit(data1[0], data1[1], data2[0], data2[1], para1);
		vector<double> para2;
		linefit(data1[0], data1[1], data3[0], data3[1], para2);
		vector<double> para3;
		linefit(data2[0], data2[1], data3[0], data3[1], para3);
		double params[3][3] = { {para1[0],para1[1],para1[2]}, {para2[0],para2[1],para2[2]}, {para3[0],para3[1],para3[2]} };
		//
		for (int pointx = dataXmin; pointx <= dataXmax; pointx++)
		{
			vector<double> yBox;
			for (int j = 0; j < 3; j++)
			{
				double tempy;
				if (params[j][2] > 0.0)
				{
					tempy = params[j][0] * (double)pointx + params[j][1];
					if ((tempy >= dataYmin) && (tempy <= dataYmax))
					{
						yBox.push_back(tempy);
					}
				}


			}
			//
			if (yBox.size() > 0)
			{
				int yBegin = ceil(*min_element(yBox.begin(), yBox.end()));
				int yEnd = floor(*max_element(yBox.begin(), yBox.end()));
				for (int pointy = yBegin; pointy <= yEnd; pointy++)
				{
					if (map[(int)pointx][(int)pointy] == 0.0)
					{
						map[(int)pointx][(int)pointy] = pointCal(data1[0], data1[1], data1[2], data2[0], data2[1], data2[2], data3[0], data3[1], data3[2], pointx, pointy);
					}

				}
			}
		}

	}



	// 释放空间
	for (int i = 0; i < numberData; i++)
	{
		delete[] sanjiaoData[i];
	}
	delete[] sanjiaoData;

}

//---------------------------------------------------------------------------------------------------
// 动态规划
//---------------------------------------------------------------------------------------------------
//
Mat dynamicProgramming(Mat im, double var)
{
	int row = im.rows;
	int col = im.cols;

	double trans = 0;
	double w_dia = 1;
	if (var > 0)
	{
		w_dia = var;
	}
	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(im, &minv, &maxv, &pt_min, &pt_max);

	//
	Mat I = Mat::zeros(row, col, CV_64FC1);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			I.at<double>(i, j) = im.at<double>(i, j) / maxv;
		}

	}
	//
	Mat paths = Mat::zeros(row, col, CV_64FC1);
	Mat l = Mat::zeros(row, col, CV_64FC1);
	double s_1;
	double s_2;
	double s_3;
	double value;
	int pos;
	//
	for (int j = 1; j < col; j++)
	{
		// 第一行
		s_1 = weight(I.at<double>(0, j - 1), I.at<double>(0, j)) + l.at<double>(0, j - 1);
		s_2 = weight(I.at<double>(1, j - 1), I.at<double>(0, j)) * w_dia + l.at<double>(1, j - 1);
		double array1[2] = { s_1,s_2 };
		value = (*min_element(array1, array1 + 2));
		pos = min_element(array1, array1 + 2) - array1 + 1;

		l.at<double>(0, j) = value;
		paths.at<double>(0, j) = pos;

		// 第二行开始
		for (int i = 1; i < row - 1; i++)
		{
			s_1 = weight(I.at<double>(i - 1, j - 1), I.at<double>(i, j)) * w_dia + l.at<double>(i - 1, j - 1);
			s_2 = weight(I.at<double>(i, j - 1), I.at<double>(i, j)) + l.at<double>(i, j - 1);
			s_3 = weight(I.at<double>(i + 1, j - 1), I.at<double>(i, j)) * w_dia + l.at<double>(i + 1, j - 1);
			double array2[3] = { s_1,s_2,s_3 };
			value = (*min_element(array2, array2 + 3));
			pos = min_element(array2, array2 + 3) - array2 + 0;

			l.at<double>(i, j) = value;
			paths.at<double>(i, j) = i + 1 + pos - 2;//i要不要+1
		}


		// 最后一行
		s_1 = weight(I.at<double>(row - 2, j - 1), I.at<double>(row - 1, j)) * w_dia + l.at<double>(row - 2, j - 1);
		s_2 = weight(I.at<double>(row - 1, j - 1), I.at<double>(row - 1, j)) + l.at<double>(row - 1, j - 1);
		double array3[2] = { s_1,s_2 };
		value = (*min_element(array3, array3 + 2));
		pos = min_element(array3, array3 + 2) - array3 + 0;

		l.at<double>(row - 1, j) = value;
		paths.at<double>(row - 1, j) = row + pos - 2;
	}

	Mat path = Mat::zeros(1, col, CV_64FC1);

	double* array4 = new double[row];
	for (int j = 0; j < row; j++)
	{
		array4[j] = l.at<double>(j, col - 1);
	}
	value = (*min_element(array4, array4 + row));
	pos = min_element(array4, array4 + row) - array4 + 0;
	delete[] array4;


	path.at<double>(0, col - 1) = pos;
	for (int i = 0; i < col - 1; i++)
	{
		int j = col - 1 - i;
		path.at<double>(0, j - 1) = paths.at<double>(path.at<double>(0, j), j);
	}
	return path;


}

//---------------------------------------------------------------------------------------------------
// 权重函数
//---------------------------------------------------------------------------------------------------
//
double weight(double ga, double gb)
{
	double w = 2 - ga - gb + 0.00001;
	return w;
}


//---------------------------------------------------------------------------------------------------
// 梯度特征提取
//---------------------------------------------------------------------------------------------------
//
Mat gradient(Mat im)
{
	int row = im.rows;
	int col = im.cols;
	Mat temp = Mat::zeros(row, col, CV_64FC1);

	for (int i = 1; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp.at<double>(i, j) = im.at<double>(i, j) - im.at<double>(i - 1, j);
		}

	}
	return temp;
}

Mat segline_template_1(int col, int row, int fineHeight, int ca_left, int ca_right,
	Mat IiCutFliFine, Mat lineRough, Mat Ii, Mat pic, Mat m_resultImg, double* label,
	string filename, string fname, double& center, Scalar color_line) {
	Mat evenIm1 = Mat::zeros(fineHeight, col, CV_64FC1);  //(2048,101)
	Mat evenIm2 = Mat::zeros(fineHeight, col, CV_64FC1);  //(2048,101)
	//// 第一次精细分割
	for (int j = 0; j < col; j++)
	{
		for (int i = 0; i < fineHeight; i++)
		{
			evenIm1.at<double>(i, j) = IiCutFliFine.at<double>((i - int(fineHeight / 2) + (int)lineRough.at<double>(0, j)), j);
		}
	}
	//replace_name_saveimg(filename, "_"+ fname + "_evenIm1_", evenIm1);
	Mat pathFine = dynamicProgramming(evenIm1, 1.1);

	Mat pathFineRestore = Mat::zeros(1, col, CV_64FC1);   //第一次精细分割，以第一次粗略分割出的图为基础走动态规划出的曲线
	for (int i = 0; i < col; i++)
	{
		pathFineRestore.at<double>(0, i) = lineRough.at<double>(0, i) + pathFine.at<double>(0, i) - int(fineHeight / 2);
	}

	vector<Point> xy2;
	for (int i = 0; i < col; i++)
	{
		if ((i > ca_right + 20) || (i < ca_left + 20))
		{
			Point dotData;
			dotData.x = (double)i;
			dotData.y = pathFineRestore.at<double>(0, i);
			xy2.push_back(dotData);
		}
	}

	Mat lineFine1 = Mat::zeros(1, col, CV_64FC1);  //第一次精细分割后多项式拟合出的曲线y值
	//Mat mat_k2 = polyfit(xy2, 4);
	Mat mat_k2 = polyfit(xy2, 3);
	//Mat mat_k2 = polyfit(xy2, 2);
	for (int i = 0; i < col; i++)
	{
		double x = (double)i;
		//double k00 = mat_k2.at<double>(4, 0);
		double k0 = mat_k2.at<double>(3, 0);
		double k1 = mat_k2.at<double>(2, 0);
		double k2 = mat_k2.at<double>(1, 0);
		double k3 = mat_k2.at<double>(0, 0);
		//double y = k00 * x * x * x * x + k0 * x * x * x +  k1 * x * x + k2 * x + k3;
		double y = k0 * x * x * x + k1 * x * x + k2 * x + k3;
		//double y =  k1 * x * x + k2 * x + k3;
		lineFine1.at<double>(0, i) = round(y);
		//circle(m_resultImg, Point(i, int(y)), 1, cv::Scalar(255, 129, 100), 1);
	}
	//imwrite(filename, m_resultImg);

	// 第二次精细分割
	for (int j = 0; j < col; j++)
	{
		for (int i = 0; i < fineHeight; i++)
		{
			evenIm2.at<double>(i, j) = Ii.at<double>((i - int(fineHeight / 2) + (int)lineFine1.at<double>(0, j)), j);
		}
	}
	pathFine = dynamicProgramming(evenIm2, 1.1);    //第二次精细分割，以第一次精细分割出的图为基础走动态规划出的曲线

	//Mat pathFineRestore = Mat::zeros(1, col, CV_64FC1);
	for (int i = 0; i < col; i++)
	{
		pathFineRestore.at<double>(0, i) = lineFine1.at<double>(0, i) + pathFine.at<double>(0, i) - int(fineHeight / 2);
	}

	vector<Point> xy3;
	for (int i = 0; i < col; i++)
	{
		//int count = 0;
		if ((i > ca_right + 20) || (i < ca_left + 20))
		{
			Point dotData;
			dotData.x = (double)i;
			dotData.y = pathFineRestore.at<double>(0, i);
			xy3.push_back(dotData);
		}
		//count = count + 1;


	}


	Mat lineFine2 = Mat::zeros(1, col, CV_64FC1);

	//Mat mat_k3 = polyfit(xy3, 4);  //第二次精细分割后多项式拟合出的曲线y值
	Mat mat_k3 = polyfit(xy3, 3);  //第二次精细分割后多项式拟合出的曲线y值
	//Mat mat_k3 = polyfit(xy3, 2);  //第二次精细分割后多项式拟合出的曲线y值
	for (int i = 0; i < col; i++)
	{
		double x = (double)i;
		//double k00 = mat_k3.at<double>(4, 0);
		double k0 = mat_k3.at<double>(3, 0);
		double k1 = mat_k3.at<double>(2, 0);
		double k2 = mat_k3.at<double>(1, 0);
		double k3 = mat_k3.at<double>(0, 0);
		//double y = k00 * x * x * x * x + k0 * x * x * x + k1 * x * x + k2 * x + k3;
		double y = k0 * x * x * x + k1 * x * x + k2 * x + k3;
		//double y = k1 * x * x + k2 * x + k3;
		lineFine2.at<double>(0, i) = round(y);
		//circle(m_resultImg, Point(i, int(y)), 1, cv::Scalar(255, 129, 100), 1);
	}



	//
	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(lineFine2, &minv, &maxv, &pt_min, &pt_max);
	double checkValue = minv;
	int centerx1 = pt_min.x;
	int centerx2 = centerx1;
	for (int i = centerx1; i < col; i++)  // 找到最小中心点的x坐标
	{
		if (lineFine2.at<double>(0, i) > checkValue)
		{
			centerx2 = i - 1;
			break;
		}
	}

	Mat imtest = Mat::zeros(1, (int)checkValue + 1, CV_64FC1);

	//
	int fixOffset = 20;
	for (int i = 0; i < (int)checkValue + 1; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (i < 20)
			{
				imtest.at<double>(0, i) = 0;
			}
			else
			{
				imtest.at<double>(0, i) = imtest.at<double>(0, i) + (double)pic.at<uint8_t>(i + fixOffset, j);
			}

		}
	}

	minMaxLoc(imtest, &minv, &maxv, &pt_min, &pt_max);
	double centery = (double)pt_max.x;





	for (int i = 0; i < col; i++)
	{
		if (pathFineRestore.at<double>(0, i) < centery + fixOffset)  // 消除中间亮线造成的影响
		{
			if (ca_left < ca_right)  // y值加fixOffset，整体坐标往下走
			{
				pathFineRestore.at<double>(0, i) = centery + fixOffset;
			}
		}
	}

	Mat xx = Mat::zeros(1, col, CV_64FC1);
	for (int i = 0;i < col; i++) xx.at<double>(0, i) = i;
	Mat pathFineRestore_bspline = Mat::zeros(1, col, CV_64FC1);
	BsplinePre(xx, pathFineRestore, xx, pathFineRestore_bspline, 100);

	for (int i = 0; i < col; i++) {
		double y = pathFineRestore_bspline.at<double>(0, i);
		if (i % 5 == 1)
		{
			circle(m_resultImg, Point(i, int(y)), 1, color_line, 1);
		}
		//label[i] = round(pathFineRestore.at<double>(0, i)) - checkValue;  //最小值为0
		label[i] = pathFineRestore_bspline.at<double>(0, i);  //最小值不需要变成0
	}

	center = round((centerx1 + centerx2) / 2);
	return pathFineRestore_bspline;
}

Mat segline_template_2(int col, int row, int fineHeight, int ca_left, int ca_right,
	Mat IiCutFliFine, Mat lineRough2, Mat lineRough3, Mat Ii, Mat pic, Mat m_resultImg, double* label,
	string filename, string fname, double& center, Scalar color_line) {



	Mat evenIm1_2 = Mat::zeros(fineHeight, col, CV_64FC1);  //(2048,101)
	Mat evenIm1_3 = Mat::zeros(fineHeight, col, CV_64FC1);  //(2048,101)
	Mat evenIm2 = Mat::zeros(fineHeight, col, CV_64FC1);  //(2048,101)
	for (int j = 0; j < col; j++)
	{
		for (int i = 0; i < fineHeight; i++)
		{
			evenIm1_2.at<double>(i, j) = IiCutFliFine.at<double>((i - int(fineHeight / 2) + (int)lineRough2.at<double>(0, j)), j);
		}
	}
	for (int j = 0; j < col; j++)
	{
		for (int i = 0; i < fineHeight; i++)
		{
			evenIm1_3.at<double>(i, j) = IiCutFliFine.at<double>((i - int(fineHeight / 2) + (int)lineRough3.at<double>(0, j)), j);
		}
	}
	replace_name_saveimg(filename, "_"+ fname + "_evenIm1_2_", evenIm1_2);
	replace_name_saveimg(filename, "_"+ fname + "_evenIm1_3_", evenIm1_3);
	Mat pathFine_2 = dynamicProgramming(evenIm1_2, 1.1);
	Mat pathFine_3 = dynamicProgramming(evenIm1_3, 1.1);

	double sum2=0, sum3=0;
	for (int i = 0;i < col;i++) sum2 += evenIm1_2.at<double>(pathFine_2.at<double>(0, i), i);
	for (int i = 0;i < col;i++) sum3 += evenIm1_3.at<double>(pathFine_3.at<double>(0, i), i);

	Mat pathFine = (sum2 > sum3) ? pathFine_2 : pathFine_3;
	Mat lineRough = (sum2 > sum3) ? lineRough2 : lineRough3;


	Mat pathFineRestore = Mat::zeros(1, col, CV_64FC1);   //第一次精细分割，以第一次粗略分割出的图为基础走动态规划出的曲线
	for (int i = 0; i < col; i++)
	{
		pathFineRestore.at<double>(0, i) = lineRough.at<double>(0, i) + pathFine.at<double>(0, i) - int(fineHeight / 2);
	}

	vector<Point> xy2;
	for (int i = 0; i < col; i++)
	{
		if ((i > ca_right + 20) || (i < ca_left + 20))
		{
			Point dotData;
			dotData.x = (double)i;
			dotData.y = pathFineRestore.at<double>(0, i);
			xy2.push_back(dotData);
		}
	}


	Mat lineFine1 = Mat::zeros(1, col, CV_64FC1);  //第一次精细分割后多项式拟合出的曲线y值

	//Mat mat_k2 = polyfit(xy2, 4);
	Mat mat_k2 = polyfit(xy2, 3);
	//Mat mat_k2 = polyfit(xy2, 2);
	for (int i = 0; i < col; i++)
	{
		double x = (double)i;
		//double k00 = mat_k2.at<double>(4, 0);
		double k0 = mat_k2.at<double>(3, 0);
		double k1 = mat_k2.at<double>(2, 0);
		double k2 = mat_k2.at<double>(1, 0);
		double k3 = mat_k2.at<double>(0, 0);
		//double y = k00 * x * x * x * x + k0 * x * x * x +  k1 * x * x + k2 * x + k3;
		double y = k0 * x * x * x + k1 * x * x + k2 * x + k3;
		//double y =  k1 * x * x + k2 * x + k3;
		lineFine1.at<double>(0, i) = round(y);
		//circle(m_resultImg, Point(i, int(y)), 1, cv::Scalar(255, 129, 100), 1);
	}
	//imwrite(filename, m_resultImg);

	// 第二次精细分割
	for (int j = 0; j < col; j++)
	{
		for (int i = 0; i < fineHeight; i++)
		{
			evenIm2.at<double>(i, j) = Ii.at<double>((i - int(fineHeight / 2) + (int)lineFine1.at<double>(0, j)), j);
		}
	}
	//imwrite(filename, Ii);
	//imwrite(filename.replace(filename.find("_evenIm1_"), 9, "_evenIm2_"), evenIm2);
	pathFine = dynamicProgramming(evenIm2, 1.1);    //第二次精细分割，以第一次精细分割出的图为基础走动态规划出的曲线

	//Mat pathFineRestore = Mat::zeros(1, col, CV_64FC1);
	for (int i = 0; i < col; i++)
	{
		pathFineRestore.at<double>(0, i) = lineFine1.at<double>(0, i) + pathFine.at<double>(0, i) - int(fineHeight / 2);
	}

	vector<Point> xy3;
	for (int i = 0; i < col; i++)
	{
		//int count = 0;
		if ((i > ca_right + 20) || (i < ca_left + 20))
		{
			Point dotData;
			dotData.x = (double)i;
			dotData.y = pathFineRestore.at<double>(0, i);
			xy3.push_back(dotData);
		}
		//count = count + 1;


	}


	Mat lineFine2 = Mat::zeros(1, col, CV_64FC1);

	//Mat mat_k3 = polyfit(xy3, 4);  //第二次精细分割后多项式拟合出的曲线y值
	Mat mat_k3 = polyfit(xy3, 3);  //第二次精细分割后多项式拟合出的曲线y值
	//Mat mat_k3 = polyfit(xy3, 2);  //第二次精细分割后多项式拟合出的曲线y值
	for (int i = 0; i < col; i++)
	{
		double x = (double)i;
		//double k00 = mat_k3.at<double>(4, 0);
		double k0 = mat_k3.at<double>(3, 0);
		double k1 = mat_k3.at<double>(2, 0);
		double k2 = mat_k3.at<double>(1, 0);
		double k3 = mat_k3.at<double>(0, 0);
		//double y = k00 * x * x * x * x + k0 * x * x * x + k1 * x * x + k2 * x + k3;
		double y = k0 * x * x * x + k1 * x * x + k2 * x + k3;
		//double y = k1 * x * x + k2 * x + k3;
		lineFine2.at<double>(0, i) = round(y);
		//circle(m_resultImg, Point(i, int(y)), 1, cv::Scalar(255, 129, 100), 1);
	}



	//
	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(lineFine2, &minv, &maxv, &pt_min, &pt_max);
	double checkValue = minv;
	int centerx1 = pt_min.x;
	int centerx2 = centerx1;
	for (int i = centerx1; i < col; i++)  // 找到最小中心点的x坐标
	{
		if (lineFine2.at<double>(0, i) > checkValue)
		{
			centerx2 = i - 1;
			break;
		}
	}

	Mat imtest = Mat::zeros(1, (int)checkValue + 1, CV_64FC1);

	//
	int fixOffset = 20;
	for (int i = 0; i < (int)checkValue + 1; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (i < 20)
			{
				imtest.at<double>(0, i) = 0;
			}
			else
			{
				imtest.at<double>(0, i) = imtest.at<double>(0, i) + (double)pic.at<uint8_t>(i + fixOffset, j);
			}

		}
	}

	minMaxLoc(imtest, &minv, &maxv, &pt_min, &pt_max);
	double centery = (double)pt_max.x;





	for (int i = 0; i < col; i++)
	{
		if (pathFineRestore.at<double>(0, i) < centery + fixOffset)  // 消除中间亮线造成的影响
		{
			if (ca_left < ca_right)  // y值加fixOffset，整体坐标往下走
			{
				pathFineRestore.at<double>(0, i) = centery + fixOffset;
			}
		}
	}

	Mat xx = Mat::zeros(1, col, CV_64FC1);
	for (int i = 0;i < col; i++) xx.at<double>(0, i) = i;
	Mat pathFineRestore_bspline = Mat::zeros(1, col, CV_64FC1);
	BsplinePre(xx, pathFineRestore, xx, pathFineRestore_bspline, 100);

	for (int i = 0; i < col; i++) {
		double y = pathFineRestore_bspline.at<double>(0, i);
		if (i % 5 == 1)
		{
			circle(m_resultImg, Point(i, int(y)), 1, color_line, 1);
		}
		//label[i] = round(pathFineRestore.at<double>(0, i)) - checkValue;  //最小值为0
		label[i] = pathFineRestore_bspline.at<double>(0, i);  //最小值不需要变成0
	}
	center = round((centerx1 + centerx2) / 2);
	return pathFineRestore_bspline;
}

//---------------------------------------------------------------------------------------------------
// 分割模块
// 三次分割的方法
// 梯度法修改为像素法作为特征关键
//---------------------------------------------------------------------------------------------------
//
void corneaSegLine(string filename, Mat pic, double* label_top, double* label_bottom,
	int cutSize, double& center_top, double& center_bottom, int fineHeight_startidx, int fineHeight)
{
	int row = pic.rows;
	int col = pic.cols;
	Mat I = Mat::zeros(row, col, CV_64FC1);
	Mat SumI = Mat::zeros(1, col, CV_64FC1); // 除去竖条纹
	Mat SumHor = Mat::zeros(1, col, CV_64FC1); // 出去横条纹
	Mat resultImg = pic.clone();
	Mat m_resultImg = Mat(resultImg.rows, resultImg.cols, CV_8UC4);
	cvtColor(resultImg, m_resultImg, COLOR_GRAY2RGB);
	// 转换为双精度
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			I.at<double>(i, j) = (double)pic.at<uint8_t>(i, j);
		}

	}
	// 竖条
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			SumI.at<double>(0, i) = SumI.at<double>(0, i) + (double)pic.at<uint8_t>(j, i);
		}
	}
	// 横条
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			SumHor.at<double>(0, i) = SumHor.at<double>(0, i) + (double)pic.at<uint8_t>(i, j);
		}

	}

	// 1.除去中央亮线模块
	int length = (round(col / 3) - 1 + 1) + (col - round(col * 2 / 3) + 1);
	double caStandard = 0;
	for (int i = 0; i < col; i++)
	{
		if (i <= round(col / 3) || (i >= round(col * 2 / 3)))
		{
			caStandard = caStandard + SumI.at<double>(0, i);

		}
	}
	caStandard = 2 * caStandard / (double)length;

	for (int i = 0; i < col / 4; i++)
	{
		SumI.at<double>(0, i) = 0;
	}
	for (int i = col * 3 / 4 - 1; i < col; i++)
	{
		SumI.at<double>(0, i) = 0;
	}
	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(SumI, &minv, &maxv, &pt_min, &pt_max);

	double ca_max = maxv;
	int ca_max_p = pt_max.x;

	int ca_left = 0;
	int ca_right = 0;

	if (ca_max < caStandard) // 没有central artifact
	{
		ca_left = round(col / 2);
		ca_right = ca_left;
	}
	else
	{
		ca_left = ca_max_p;
		ca_right = ca_max_p;
		for (int i = 1; i <= round(col / 3); i++)
		{
			if (SumI.at<double>(0, ca_max_p - i) > caStandard)
			{
				ca_left = ca_max_p - i;
			}
			if (SumI.at<double>(0, ca_max_p + i) > caStandard)
			{
				ca_right = ca_max_p + i;
			}

		}

	}

	// 2.除去横亮条纹
	Mat Ii = Mat::zeros(row, col, CV_64FC1);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			Ii.at<double>(i, j) = I.at<double>(i, j) - SumHor.at<double>(0, i) / (double)col;
		}
	}

	minMaxLoc(Ii, &minv, &maxv, &pt_min, &pt_max);
	double IiMin = minv;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			Ii.at<double>(i, j) = Ii.at<double>(i, j) + abs(minv);
		}
	}

	// 除去竖条亮纹
	auto mean_val = mean(Ii);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if ((j > ca_left) && (j < ca_right))
			{
				//Ii.at<double>(i, j) = 0;
				//Ii.at<double>(i, j) = caStandard / col;
				Ii.at<double>(i, j) = mean_val[0];
			}
		}
	}


	// 3.
	Mat IiCutFliRough = Mat::zeros(row, col, CV_64FC1);
	Mat IiCutFliFine = Mat::zeros(row, col, CV_64FC1);

	replace_name_saveimg(filename, "_Ii_", Ii);


	//imwrite(filename, Ii);
	blur(Ii, IiCutFliRough, cv::Size(11, 11));
	blur(Ii, IiCutFliFine, cv::Size(5, 5));

	/*IiCutFliRough = gradient(IiCutFliRough);
	IiCutFliFine = gradient(IiCutFliFine);*/



	// 中央cutsize粗略分割
	//int cutSize = 650;
	length = col - 0 - cutSize - cutSize;
	Mat roughRegion = Mat::zeros(row, length, CV_64FC1);
	for (int i = 0; i < row; i++)
	{
		for (int j = cutSize; j <= col - cutSize; j++)
		{
			roughRegion.at<double>(i, j - cutSize) = IiCutFliRough.at<double>(i, j);
		}

	}

	double minVal, maxVal;
	int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
	minMaxIdx(roughRegion, &minVal, &maxVal, minIdx, maxIdx);
	Mat roughRegion_uint = Mat::zeros(row, length, CV_8UC1);
	roughRegion.convertTo(roughRegion_uint, CV_8UC1, 255.0 / (maxVal - minVal), 0);


	// 膨胀
	Mat mask;
	vector<vector<Point>> contours;
	double grayth = threshold(roughRegion_uint, mask, 0, 255, THRESH_OTSU);  // 自适应阈值
	replace_name_saveimg(filename, "_maskori_", mask);

	//// 开运算去除噪点
	//Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
	//morphologyEx(mask, mask, MORPH_OPEN, element);
	mask.rowRange(0, 150) = 0;


	Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(100, 50));  // 膨胀
	//Mat mask_close, mask_open, mask_tmp;
	dilate(mask, mask, element);

	//找轮廓
	findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	sort(contours.begin(), contours.end(), Contour_Area);
	const int contours_size = contours.size();
	for (int i = 0; i < contours_size - 1;i++) { // 找出最大轮廓
		contours.pop_back();
	}

	mask.setTo(0);  //值设为0，填充黑色
	cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);  //值设为255，填充白色

	//morphologyEx(mask_tmp, mask_close, MORPH_CLOSE, element);
	//morphologyEx(mask_tmp, mask_open, MORPH_OPEN, element);
	replace_name_saveimg(filename, "_mask_", mask);
	vector<Point> CorneaFC_out = BoundaryDetect_pixel(mask, 100, cutSize);  //第一次粗略分割
	for (auto xy : CorneaFC_out) {
		circle(m_resultImg, xy, 1, cv::Scalar(255, 0, 0), 1);
	}

	Mat CorneaFC_out_mat = (Mat{ CorneaFC_out }).reshape(1, 0).t(); // vector转Mat
	minMaxIdx(CorneaFC_out_mat.rowRange(1,2), &minVal, &maxVal, minIdx, maxIdx);
	if (maxVal - minVal <= 10) {
		// 第二种方法
		Mat roughPath = Mat::zeros(1, length, CV_64FC1);  //第一次粗略分割，以中央cutsize区域的图为基础走动态规划出的曲线
		roughPath = dynamicProgramming(roughRegion, 1.0);
		for (int i = cutSize; i <= col - cutSize; i++) {
			CorneaFC_out[i- cutSize] = Point(i, roughPath.at<double>(0, i - cutSize));
			circle(m_resultImg, Point(i, roughPath.at<double>(0, i - cutSize)), 1, cv::Scalar(0, 255, 255), 1);
		}
	}

	//replace_name_saveimg(filename, "_roughRegion_", m_resultImg);


	Mat lineRough = Mat::zeros(1, col, CV_64FC1);  //第一次粗略分割后多项式拟合出的曲线y值
	Mat lineRough2 = Mat::zeros(1, col, CV_64FC1);  //上界线往下平移fineHeight_startidx个像素
	Mat lineRough3 = Mat::zeros(1, col, CV_64FC1);  //上界线往上平移fineHeight_startidx个像素

	//Mat mat_k1 = polyfit(CorneaFC_out, 4);
	Mat mat_k1 = polyfit(CorneaFC_out, 3);
	//Mat mat_k1 = polyfit(CorneaFC_out, 2);
	for (int i = 0; i < col; i++)
	{
		double x = (double)i;
		//double k00 = mat_k1.at<double>(4, 0);
		double k0 = mat_k1.at<double>(3, 0);
		double k1 = mat_k1.at<double>(2, 0);
		double k2 = mat_k1.at<double>(1, 0);
		double k3 = mat_k1.at<double>(0, 0);
		//double y = k00 * x * x * x * x + k0 * x * x * x + k1 * x * x + k2 * x + k3;
		double y = k0 * x * x * x + k1 * x * x + k2 * x + k3;
		//double y = k1 * x * x + k2 * x + k3;
		lineRough.at<double>(0, i) = round(y);
		//circle(m_resultImg, Point(i, int(y)), 1, cv::Scalar(0, 0, 255), 1);
	}
	//imwrite(filename_tmp, m_resultImg);


	// 精细分割
	Mat pathFineRestore_top = segline_template_1(col, row, fineHeight, ca_left, ca_right, IiCutFliFine, lineRough, Ii, pic, m_resultImg, label_top,
		filename, "top", center_top, Scalar(0, 0, 255, 100)); // 分割上界线
	lineRough2 = pathFineRestore_top + fineHeight_startidx;
	lineRough3 = pathFineRestore_top - fineHeight_startidx;
	//for (int i = 0; i < col; i++) {
	//	lineRough2.at<double>(0, i) = round(pathFineRestore_top.at<double>(0, i) + fineHeight_startidx);
	//	//circle(m_resultImg, Point(i, pathFineRestore_top.at<double>(0, i)), 1, cv::Scalar(255, 0, 255), 1);
	//	//circle(m_resultImg, Point(i, lineRough2.at<double>(0, i)), 1, cv::Scalar(255, 255, 0), 1);
	//}

	Mat pathFineRestore_bottom = segline_template_2(col, row, fineHeight, ca_left, ca_right, IiCutFliFine, lineRough2, lineRough3, Ii, pic, m_resultImg, label_bottom,
		filename, "bottom", center_bottom, Scalar(0, 255, 0, 100)); // 分割下界线

	if (mean(pathFineRestore_top).val[0] > mean(pathFineRestore_bottom).val[0]) { // 交换两条线
		for (int i = 0;i < col;i++) {
			double tmp = label_top[i];
			label_top[i] = label_bottom[i];
			label_bottom[i] = tmp;
		}
	}
	imwrite(filename, m_resultImg);
}


Mat cal_z(double r_x, double r_y, double r_z, double r, double dimxy) {
	Mat cornea_fit_sphere = Mat::zeros(dimxy, dimxy, CV_64FC1);
	for (int i = 0; i < dimxy; i++) {
		for (int j = 0; j < dimxy; j++) {
			cornea_fit_sphere.at<double>(i, j) = sqrt(abs(r * r - (i - dimxy/2 - r_x) * (i - dimxy / 2 - r_x) 
				- (j - dimxy / 2 - r_y) * (j - dimxy / 2 - r_y))) + r_z;
		}
	}
	return cornea_fit_sphere;
}

//
// 曲率计算，返回圆心位置
//
double fitspherebYTandReturn(vector<double>Rx, vector<double>Ry, vector<double>Rz, vector<double>& center)
{
	double x=0, y=0,z = 0, x_avr = 0, y_avr = 0, z_avr = 0, xx_avr = 0, yy_avr = 0, zz_avr = 0, xy_avr = 0, xz_avr = 0, yz_avr = 0,
		xxx_avr = 0, xxy_avr = 0, xxz_avr = 0, xyy_avr = 0, xzz_avr = 0, yyy_avr = 0, yyz_avr = 0, yzz_avr = 0, zzz_avr = 0, r_x, r_y, r_z;
	double num_points = Rx.size();
	for (int i = 0; i < num_points;i++) {
		x = Rx[i];
		y = Ry[i];
		z = Rz[i];
		x_avr += x;
		y_avr += y;
		z_avr += z;
		xx_avr += x * x;
		yy_avr += y * y;
		zz_avr += z * z;
		xy_avr += x * y;
		xz_avr += x * z;
		yz_avr += y * z;
		xxx_avr += x * x * x;
		xxy_avr += x * x * y;
		xxz_avr += x * x * z;
		xyy_avr += x * y * y;
		xzz_avr += x * z * z;
		yyy_avr += y * y * y;
		yyz_avr += y * y * z;
		yzz_avr += y * z * z;
		zzz_avr += z * z * z;
	}
	x_avr /= num_points;
	y_avr /= num_points;
	z_avr /= num_points;
	xx_avr /= num_points;
	yy_avr /= num_points;
	zz_avr /= num_points;
	xy_avr /= num_points;
	xz_avr /= num_points;
	yz_avr /= num_points;
	xxx_avr /= num_points;
	xxy_avr /= num_points;
	xxz_avr /= num_points;
	xyy_avr /= num_points;
	xzz_avr /= num_points;
	yyy_avr /= num_points;
	yyz_avr /= num_points;
	yzz_avr /= num_points;
	zzz_avr /= num_points;
	Mat A = (Mat_<double>(3, 3) << xx_avr - x_avr * x_avr, xy_avr - x_avr * y_avr, xz_avr - x_avr * z_avr,
			xy_avr - x_avr * y_avr, yy_avr - y_avr * y_avr, yz_avr - y_avr * z_avr,
			xz_avr - x_avr * z_avr, yz_avr - y_avr * z_avr, zz_avr - z_avr * z_avr);//直接赋初始值的方法
	Mat b = (Mat_<double>(3, 1) << xxx_avr - x_avr * xx_avr + xyy_avr - x_avr * yy_avr + xzz_avr - x_avr * zz_avr,
			xxy_avr - y_avr * xx_avr + yyy_avr - y_avr * yy_avr + yzz_avr - y_avr * zz_avr,
			xxz_avr - z_avr * xx_avr + yyz_avr - z_avr * yy_avr + zzz_avr - z_avr * zz_avr);//直接赋初始值的方法
	b = b / 2;
	Mat center_mat(3, 1, CV_64FC1);
	center_mat = (A.t() * A).inv() * A.t() * b; //inv求逆矩阵，mat_k shape:[x_num, 1]
	r_x = center_mat.at<double>(0, 0), r_y = center_mat.at<double>(1, 0), r_z = center_mat.at<double>(2, 0);
	center.push_back(r_x);
	center.push_back(r_y);
	center.push_back(r_z);
	double r2 = xx_avr - 2 * r_x * x_avr + r_x * r_x + yy_avr - 2 * r_y * y_avr + r_y * r_y + zz_avr - 2 * r_z * z_avr + r_z * r_z;
	return sqrt(r2); // 818
}




//
// 绘制色标，白色在下面
//
Mat colorBar(int dimy, double value1, double value2)
{
	int dimBarx = dimy;
	int dimBary = 400;
	Mat colorBarMatrix = Mat::zeros(dimBarx, dimBary, CV_64FC1);
	Mat maskBar = Mat::zeros(dimBarx, dimBary, CV_8UC1);

	double start = 0.0;
	int begin = 200;
	int end = dimy - 200 + 1;
	int sizebar = (end - begin) / 10;

	double dx = (value2 - value1) / (end - 1 - begin);

	for (int i = begin; i <= end; ++i)
	{
		for (int j = 20; j <= 100; ++j)
		{
			colorBarMatrix.at<double>(i, j) = value2 - value1 - dx * start;
		}
		start = start + 1.0;
	}

	Mat displayBar1;
	Mat displayBar2;
	colorBarMatrix.convertTo(displayBar1, CV_8UC1, 255.0 / (value2 - value1), 0);
	//ColorMapNew picNew;
	applyColorMapYT(displayBar1, displayBar2, cv::COLORMAP_PARULA);
	Point p1 = Point(20, begin);
	Point p2 = Point(100, begin);
	Point p3 = Point(20, end);
	Point p4 = Point(100, end);

	line(maskBar, p1, p2, 255, 2, 8);
	line(maskBar, p3, p4, 255, 2, 8);
	line(maskBar, p1, p3, 255, 2, 8);
	line(maskBar, p2, p4, 255, 2, 8);

	Point dotl1 = Point(95, begin + sizebar * 1);
	Point dotr1 = Point(100, begin + sizebar * 1);
	line(maskBar, dotl1, dotr1, 255, 2, 8);
	Point dotl2 = Point(95, begin + sizebar * 2);
	Point dotr2 = Point(100, begin + sizebar * 2);
	line(maskBar, dotl2, dotr2, 255, 2, 8);
	Point dotl3 = Point(95, begin + sizebar * 3);
	Point dotr3 = Point(100, begin + sizebar * 3);
	line(maskBar, dotl3, dotr3, 255, 2, 8);
	Point dotl4 = Point(95, begin + sizebar * 4);
	Point dotr4 = Point(100, begin + sizebar * 4);
	line(maskBar, dotl4, dotr4, 255, 2, 8);
	Point dotl5 = Point(95, begin + sizebar * 5);
	Point dotr5 = Point(100, begin + sizebar * 5);
	line(maskBar, dotl5, dotr5, 255, 2, 8);
	Point dotl6 = Point(95, begin + sizebar * 6);
	Point dotr6 = Point(100, begin + sizebar * 6);
	line(maskBar, dotl6, dotr6, 255, 2, 8);
	Point dotl7 = Point(95, begin + sizebar * 7);
	Point dotr7 = Point(100, begin + sizebar * 7);
	line(maskBar, dotl7, dotr7, 255, 2, 8);
	Point dotl8 = Point(95, begin + sizebar * 8);
	Point dotr8 = Point(100, begin + sizebar * 8);
	line(maskBar, dotl8, dotr8, 255, 2, 8);
	Point dotl9 = Point(95, begin + sizebar * 9);
	Point dotr9 = Point(100, begin + sizebar * 9);
	line(maskBar, dotl9, dotr9, 255, 2, 8);

	double number[11] = { 0 };
	double dx2 = (value2 - value1) / 10;
	for (int i = 0; i < 11; i++)
	{
		number[i] = round((value2 - dx2 * i) * 1000) / 1000;
	}

	int fontface = FONT_HERSHEY_PLAIN;
	double fontscale = 3;
	int thickness = 2;
	// 中心
	string textNumber = to_string(float(number[0])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[1])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 1), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[2])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 2), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[3])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 3), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[4])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 4), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[5])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 5), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[6])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 6), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[7])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 7), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[8])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 8), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[9])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 9), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[10])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 10), fontface, fontscale, Scalar::all(0), thickness, 8);
	displayBar2.setTo(0, maskBar);
	return displayBar2;
}



//
// 绘制色标，白色在中间
//
Mat colorBar1(int dimy, double value1, double value2)
{
	int dimBarx = dimy;
	int dimBary = 400;
	Mat colorBarMatrix = Mat::zeros(dimBarx, dimBary, CV_64FC1);
	Mat maskBar = Mat::zeros(dimBarx, dimBary, CV_8UC1);

	for (int i = 0; i < dimBarx; i++)
	{
		for (int j = 0; j < dimBary; j++)
		{
			colorBarMatrix.at<double>(i, j) = value2;
		}
	}

	double start = 0.0;
	int begin = 200;
	int end = dimy - 200 + 1;
	int sizebar = (end - begin) / 10;

	double dx = (value2 - value1) / (end - 1 - begin);

	for (int i = begin; i <= end; ++i)
	{
		for (int j = 20; j <= 100; ++j)
		{
			colorBarMatrix.at<double>(i, j) = value2 - value1 - dx * start;
		}
		start = start + 1.0;
	}

	Mat displayBar1;
	Mat displayBar2;
	colorBarMatrix.convertTo(displayBar1, CV_8UC1, 255.0 / (value2 - value1), 0);
	//ColorMapNew picNew;
	applyColorMapYT(displayBar1, displayBar2, cv::COLORMAP_MAGMA);
	Point p1 = Point(20, begin);
	Point p2 = Point(100, begin);
	Point p3 = Point(20, end);
	Point p4 = Point(100, end);

	line(maskBar, p1, p2, 255, 2, 8);
	line(maskBar, p3, p4, 255, 2, 8);
	line(maskBar, p1, p3, 255, 2, 8);
	line(maskBar, p2, p4, 255, 2, 8);

	Point dotl1 = Point(95, begin + sizebar * 1);
	Point dotr1 = Point(100, begin + sizebar * 1);
	line(maskBar, dotl1, dotr1, 255, 2, 8);
	Point dotl2 = Point(95, begin + sizebar * 2);
	Point dotr2 = Point(100, begin + sizebar * 2);
	line(maskBar, dotl2, dotr2, 255, 2, 8);
	Point dotl3 = Point(95, begin + sizebar * 3);
	Point dotr3 = Point(100, begin + sizebar * 3);
	line(maskBar, dotl3, dotr3, 255, 2, 8);
	Point dotl4 = Point(95, begin + sizebar * 4);
	Point dotr4 = Point(100, begin + sizebar * 4);
	line(maskBar, dotl4, dotr4, 255, 2, 8);
	Point dotl5 = Point(95, begin + sizebar * 5);
	Point dotr5 = Point(100, begin + sizebar * 5);
	line(maskBar, dotl5, dotr5, 255, 2, 8);
	Point dotl6 = Point(95, begin + sizebar * 6);
	Point dotr6 = Point(100, begin + sizebar * 6);
	line(maskBar, dotl6, dotr6, 255, 2, 8);
	Point dotl7 = Point(95, begin + sizebar * 7);
	Point dotr7 = Point(100, begin + sizebar * 7);
	line(maskBar, dotl7, dotr7, 255, 2, 8);
	Point dotl8 = Point(95, begin + sizebar * 8);
	Point dotr8 = Point(100, begin + sizebar * 8);
	line(maskBar, dotl8, dotr8, 255, 2, 8);
	Point dotl9 = Point(95, begin + sizebar * 9);
	Point dotr9 = Point(100, begin + sizebar * 9);
	line(maskBar, dotl9, dotr9, 255, 2, 8);



	double number[11] = { 0 };
	double dx2 = (value2 - value1) / 10;
	for (int i = 0; i < 11; i++)
	{
		number[i] = round((value2 - dx2 * i) * 1000) / 1000;
	}

	int fontface = FONT_HERSHEY_PLAIN;
	double fontscale = 3;
	int thickness = 2;
	// 中心
	string textNumber = to_string(float(number[0])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[1])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 1), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[2])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 2), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[3])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 3), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[4])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 4), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[5])).substr(0, 5);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 5), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[6])).substr(0, 6);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 6), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[7])).substr(0, 6);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 7), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[8])).substr(0, 6);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 8), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[9])).substr(0, 6);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 9), fontface, fontscale, Scalar::all(0), thickness, 8);
	textNumber = to_string(float(number[10])).substr(0, 6);
	putText(displayBar2, textNumber, Point(110, 20 + begin + sizebar * 10), fontface, fontscale, Scalar::all(0), thickness, 8);


	displayBar2.setTo(0, maskBar);
	return displayBar2;

}

// 绘制三维图像时用到的颜色显示
void colorToParula(double z, double num1, double num2, vector<int>& color)
{
	static const float r[] = { 1.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.01563f,  0.03125f,  0.04688f,  0.06250f,  
		0.07813f,  0.09375f,  0.10938f,  0.12500f,  0.14063f,  0.15625f,  0.17188f,  0.18750f,  0.20313f,  0.21875f,  0.23438f,  0.25000f,  0.26563f,  0.28125f,  0.29688f,  0.31250f,  0.32813f,  
		0.34375f,  0.35938f,  0.37500f,  0.39063f,  0.40625f,  0.42188f,  0.43750f,  0.45313f,  0.46875f,  0.48438f,  0.50000f,  0.51563f,  0.53125f,  0.54688f,  0.56250f,  0.57813f,  0.59375f,  
		0.60938f,  0.62500f,  0.64063f,  0.65625f,  0.67188f,  0.68750f,  0.70313f,  0.71875f,  0.73438f,  0.75000f,  0.76563f,  0.78125f,  0.79688f,  0.81250f,  0.82813f,  0.84375f,  0.85938f,  
		0.87500f,  0.89063f,  0.90625f,  0.92188f,  0.93750f,  0.95313f,  0.96875f,  0.98438f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  0.98438f,  0.96875f,  0.95313f,  0.93750f,  0.92188f,  0.90625f,  0.89063f,  0.87500f,  0.85938f,  0.84375f,  0.82813f,  0.81250f,  
		0.79688f,  0.78125f,  0.76563f,  0.75000f,  0.73438f,  0.71875f,  0.70313f,  0.68750f,  0.67188f,  0.65625f,  0.64063f,  0.62500f,  0.60938f,  0.59375f,  0.57813f,  0.56250f,  0.54688f,  
		0.53125f,  0.51563f,  0.50000f };
	static const float g[] = { 1.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.01563f,  0.03125f,  0.04688f,  0.06250f,  0.07813f,  0.09375f,  0.10938f,  0.12500f,  0.14063f,  0.15625f,  0.17188f,  0.18750f,  0.20313f,  0.21875f,  0.23438f,  0.25000f,  0.26563f,  
		0.28125f,  0.29688f,  0.31250f,  0.32813f,  0.34375f,  0.35938f,  0.37500f,  0.39063f,  0.40625f,  0.42188f,  0.43750f,  0.45313f,  0.46875f,  0.48438f,  0.50000f,  0.51563f,  0.53125f,  
		0.54688f,  0.56250f,  0.57813f,  0.59375f,  0.60938f,  0.62500f,  0.64063f,  0.65625f,  0.67188f,  0.68750f,  0.70313f,  0.71875f,  0.73438f,  0.75000f,  0.76563f,  0.78125f,  0.79688f,  
		0.81250f,  0.82813f,  0.84375f,  0.85938f,  0.87500f,  0.89063f,  0.90625f,  0.92188f,  0.93750f,  0.95313f,  0.96875f,  0.98438f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  0.98438f,  0.96875f,  0.95313f,  0.93750f,  0.92188f,  0.90625f,  0.89063f,  0.87500f,  
		0.85938f,  0.84375f,  0.82813f,  0.81250f,  0.79688f,  0.78125f,  0.76563f,  0.75000f,  0.73438f,  0.71875f,  0.70313f,  0.68750f,  0.67188f,  0.65625f,  0.64063f,  0.62500f,  0.60938f,  
		0.59375f,  0.57813f,  0.56250f,  0.54688f,  0.53125f,  0.51563f,  0.50000f,  0.48438f,  0.46875f,  0.45313f,  0.43750f,  0.42188f,  0.40625f,  0.39063f,  0.37500f,  0.35938f,  0.34375f,  
		0.32813f,  0.31250f,  0.29688f,  0.28125f,  0.26563f,  0.25000f,  0.23438f,  0.21875f,  0.20313f,  0.18750f,  0.17188f,  0.15625f,  0.14063f,  0.12500f,  0.10938f,  0.09375f,  0.07813f,  
		0.06250f,  0.04688f,  0.03125f,  0.01563f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f };
	static const float b[] = { 1.00000f,  0.53125f,  0.54688f,  0.56250f,  0.57813f,  0.59375f,  0.60938f,  0.62500f,  0.64063f,  0.65625f,  0.67188f,  0.68750f,  0.70313f,  0.71875f,  0.73438f,  
		0.75000f,  0.76563f,  0.78125f,  0.79688f,  0.81250f,  0.82813f,  0.84375f,  0.85938f,  0.87500f,  0.89063f,  0.90625f,  0.92188f,  0.93750f,  0.95313f,  0.96875f,  0.98438f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  
		1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  1.00000f,  0.98438f,  0.96875f,  0.95313f,  0.93750f,  
		0.92188f,  0.90625f,  0.89063f,  0.87500f,  0.85938f,  0.84375f,  0.82813f,  0.81250f,  0.79688f,  0.78125f,  0.76563f,  0.75000f,  0.73438f,  0.71875f,  0.70313f,  0.68750f,  0.67188f,  
		0.65625f,  0.64063f,  0.62500f,  0.60938f,  0.59375f,  0.57813f,  0.56250f,  0.54688f,  0.53125f,  0.51563f,  0.50000f,  0.48438f,  0.46875f,  0.45313f,  0.43750f,  0.42188f,  0.40625f,  
		0.39063f,  0.37500f,  0.35938f,  0.34375f,  0.32813f,  0.31250f,  0.29688f,  0.28125f,  0.26563f,  0.25000f,  0.23438f,  0.21875f,  0.20313f,  0.18750f,  0.17188f,  0.15625f,  0.14063f,  
		0.12500f,  0.10938f,  0.09375f,  0.07813f,  0.06250f,  0.04688f,  0.03125f,  0.01563f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f, 
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  0.00000f,  
		0.00000f,  0.00000f,  0.00000f };
	int ratio = floor(256 * (z - num2) / (num1 - num2));
	int red = floor(255 * r[ratio]);
	int green = floor(255 * g[ratio]);
	int blue = floor(255 * b[ratio]);
	color.push_back(red);
	color.push_back(green);
	color.push_back(blue);
}

// 径向公式的计算
float zernike_radial(float r, float n, float m)
{
	float radial;
	if (n == m)
	{
		radial = pow(r, n);
	}
	else if (((n - m) - 2) < 0.000001)
	{
		radial = n * zernike_radial(r, n, n) - (n - 1) * zernike_radial(r, n - 2, n - 2);
	}
	else
	{
		float H3 = (-4 * ((m + 4) - 2) * ((m + 4) - 3)) / ((n + (m + 4) - 2) * (n - (m + 4) + 4));
		float H2 = (H3 * (n + (m + 4)) * (n - (m + 4) + 2)) / (4 * ((m + 4) - 1)) + ((m + 4) - 2);
		float H1 = ((m + 4) * ((m + 4) - 1) / 2) - (m + 4) * H2 + (H3 * (n + (m + 4) + 2) * (n - (m + 4))) / (8);
		radial = H1 * zernike_radial(r, n, m + 4) + (H2 + H3 / pow(r, 2)) * zernike_radial(r, n, m + 2);
	}

	return radial;
}

// zernike多项式
float zernike(float r, float t, float n, float m)
{
	float zern;
	if (m < 0)
	{
		zern = -zernike_radial(r, n, -m) * sin(m * t);
	}
	else
	{
		zern = zernike_radial(r, n, m) * cos(m * t);
	}
	return zern;
}

// 处理圆外部分
Mat elliptical_crop(Mat im, float crop_frac)
{
	Mat cropped_im = im;
	float col = (double)im.cols;
	float center_x = (col - 1) / 2;
	float center_y = (col - 1) / 2;
	float radius_x = (col - center_x) * crop_frac;
	float radius_y = (col - center_y) * crop_frac;
	//
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (sqrt(pow(((float)i - center_y), 2) / pow(radius_y, 2) + pow(((float)j - center_x), 2) / pow(radius_x, 2)) > 1)
			{
				cropped_im.at<float>(i, j) = 0;
			}
		}
	}

	//
	return cropped_im;
}

//
void zernikeMat(Mat im, float indice[36][2], float* zernikeMatrix, int nZernike)
{
	int col = im.cols;
	int sizeZernike = col;
	Mat x = Mat::zeros(col, col, CV_32FC1);
	Mat y = Mat::zeros(col, col, CV_32FC1);
	Mat r = Mat::zeros(col, col, CV_32FC1);
	Mat t = Mat::zeros(col, col, CV_32FC1);

	int size = (col - 1) / 2;
	float delta = 1.0 / (float)size;
	//
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < col; j++)
		{
			float rx = 0 - delta * (float)size + (j - 1) * delta;
			float ry = 0 + delta * (float)size - (i - 1) * delta;

			x.at<float>(i, j) = rx;
			y.at<float>(i, j) = ry;

			r.at<float>(i, j) = sqrt(rx * rx + ry * ry);
			t.at<float>(i, j) = atan2(rx, ry);
		}
	}
	for (int i = 0; i < nZernike; i++)
	{
		Mat imNew = Mat::zeros(col, col, CV_32FC1);
		for (int j = 0; j < col; j++)
		{
			for (int k = 0; k < col; k++)
			{

				imNew.at<float>(j, k) = zernike(r.at<float>(j, k), t.at<float>(j, k), indice[i][0], indice[i][1]);


			}
		}
		imNew = elliptical_crop(imNew, 1.0);
		for (int j = 0; j < col; j++)
		{
			for (int k = 0; k < col; k++)
			{
				if (isnan(imNew.at<float>(j, k)))
				{
					zernikeMatrix[i * sizeZernike * sizeZernike + j * sizeZernike + k] = 0.0;
				}
				else
				{
					zernikeMatrix[i * sizeZernike * sizeZernike + j * sizeZernike + k] = imNew.at<float>(j, k);
				}

			}
		}
	}
}

// 计算系数
void zernikeCoff(Mat im, float* zernikeMatrix, double* coffZernike, int nZernike)
{
	int col = im.cols;
	int sizeZernike = col;
	Mat im_reshaped_matrix = Mat::zeros(col * col, 1, CV_64FC1); //被拟合的图片
	Mat z_reshaped_matrix = Mat::zeros(nZernike, col * col, CV_64FC1); //36层zernike矩阵
	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < col; j++)
		{
			im_reshaped_matrix.at<double>(i * col + j, 0) = im.at<double>(i, j);
		}
	}
	for (int i = 0; i < nZernike; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int k = 0; k < col; k++)
			{
				if (isnan(zernikeMatrix[i * sizeZernike * sizeZernike + j * sizeZernike + k]))
				{
					z_reshaped_matrix.at<double>(i, j * col + k) = 0;
				}
				else
				{
					z_reshaped_matrix.at<double>(i, j * col + k) = (float)zernikeMatrix[i * sizeZernike * sizeZernike + j * sizeZernike + k];
				}

			}
		}
	}
	Mat dst1 = z_reshaped_matrix * im_reshaped_matrix;
	Mat dst2 = z_reshaped_matrix * z_reshaped_matrix.t();
	Mat dst3 = dst2.inv();
	Mat dst = dst3 * dst1;
	for (int i = 0; i < nZernike; i++)
	{
		coffZernike[i] = dst.at<double>(i, 0);
	}

}

// 对图像做zernike多项式拟合，cuda传入版本
void mapZernikeCUDA(double** map, int realSize, int plotSize)
{
	Mat imRaw = Mat::zeros(plotSize, plotSize, CV_64FC1);
	for (int i = 0; i < plotSize; i++)
	{
		for (int j = 0; j < plotSize; j++)
		{
			imRaw.at<double>(i, j) = map[i][j];
			map[i][j] = 0.0;
		}
	}


	double rNormal = (double)realSize;
	double rCenter = (double)(plotSize - 1) / 2.0;
	for (int i = 0; i < plotSize; i++)
	{
		for (int j = 0; j < plotSize; j++)
		{
			double rx = ((double)i - rCenter) / rNormal;
			double ry = ((double)j - rCenter) / rNormal;
			double rxy = rx * rx + ry * ry;
			if (rxy > 1)
			{
				imRaw.at<double>(i, j) = 0.0;
			}
		}
	}
	// plotsize传输过来的图像尺寸
	// realsize做zernike拟合的图像的半径
	int sizeZernike = realSize * 2 + 1;// 做zernike拟合的图像尺寸
	int patch = (plotSize - sizeZernike) * 0.5;// 每个方向多出来的距离
	Mat im = Mat::zeros(sizeZernike, sizeZernike, CV_64FC1); // 拟合尺寸的图像
	Mat imFinal = Mat::zeros(plotSize, plotSize, CV_64FC1);//最终尺寸的图像
	for (int i = 0; i < sizeZernike; i++)
	{
		for (int j = 0; j < sizeZernike; j++)
		{
			im.at<double>(i, j) = imRaw.at<double>(i + patch, j + patch);
		}
	}
	int nZernike = 36;
	// 36项zernike多项式1-7的m，n
	float indice[36][2] = {
		{0,0},{1,-1},{1,1},{2,-2},{2,0},{2,2},{3,-3},{3,-1},{3,1},{3,3},
		{4,-4},{4,-2},{4,0},{4,2},{4,4},
		{5,-5},{5,-3},{5,-1},{5,1},{5,3},{5,5},
		{6,-6},{6,-4},{6,-2},{6,0},{6,2},{6,4},{6,6},
		{7,-7},{7,-5},{7,-3},{7,-1},{7,1},{7,3},{7,5},{7,7}
	};
	double* coffZernike = new double[nZernike];

	// cuda相关
	int numElements = nZernike * sizeZernike * sizeZernike;//一维矩阵大小
	size_t size = numElements * sizeof(float);
	float* h_zernikeMatrix = (float*)malloc(size);

	// 计算出36张基地的矩阵
	float second1 = (float)clock();
	//zernikeMat(im, indice, h_zernikeMatrix, nZernike); // cpu  10.144s
	zernike_fit_cuda(nZernike, sizeZernike, h_zernikeMatrix); //gpu  4s
	float second2 = ((float)clock() - second1) / 1000;
	cout << "time: " << second2 << endl;
	// 拟合出36个基地矩阵的系数
	zernikeCoff(im, h_zernikeMatrix, coffZernike, nZernike);
	// 重建面型
	float second3 = (float)clock();
	for (int i = 0; i < nZernike; i++)
	{
		for (int j = 0; j < sizeZernike; j++)
		{
			for (int k = 0; k < sizeZernike; k++)
			{
				map[j + patch][k + patch] = map[j + patch][k + patch] + coffZernike[i] * h_zernikeMatrix[i * sizeZernike * sizeZernike + j * sizeZernike + k];
			}
		}
	}
	float second4 = ((float)clock() - second3) / 1000;
	cout << "time: " << second4 << endl;
	delete[] coffZernike;
	cudaFree(h_zernikeMatrix);
}


// 计算角膜曲率值
double RcalCornea(double** map, double** zernike_map, int dimx, int dimy, double width, double depth, int nImgLength, double rSize, int colorbarPos,
	string path, string fname, double R1, bool if3D)
{
	//double R6 = floor(3.75 / (width / (double)nImgLength));
	// 设置角膜地形图和巩膜地形图的范围
	double Rcornea = floor(R1 / (width / (double)nImgLength));

	cout << Rcornea << endl;
	//cout << dimx << endl;

	// 接受指针中的二维数据
	Mat pic = Mat::zeros(dimx, dimy, CV_64FC1);
	for (int i = 0; i < dimx; ++i)
	{
		for (int j = 0; j < dimy; ++j)
		{
			/*double temp = 0;*/
			pic.at<double>(i, j) = map[i][j];

		}
	}

	//double minVal, maxVal;
	//int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
	//minMaxIdx(pic, &minVal, &maxVal, minIdx, maxIdx);
	//Mat display1;
	//pic.convertTo(display1, CV_8UC1, 255.0 / (maxVal - minVal), 0);


	// 简单滤波操作
	/*int filterSize = 11;
	blur(pic, pic, cv::Size(filterSize, filterSize));*/
	//mapZernike(pic, (int)Rcornea, dimx);
	//
	int nn = dimx;
	int mm = dimy;
	int startx = floor(dimx / 2);
	int starty = floor(dimy / 2);
	/*double centerx = ceil((double)dimx / 2);
	double centery = ceil((double)dimy / 2);*/
	Mat x = Mat::zeros(nn, mm, CV_64FC1);
	Mat y = Mat::zeros(nn, mm, CV_64FC1);
	/*Mat th = Mat::zeros(nn, mm, CV_64FC1);
	Mat R = Mat::zeros(nn, mm, CV_64FC1);*/
	for (int i = 0; i < nn; i++)
	{
		for (int j = 0; j < mm; j++)
		{
			x.at<double>(i, j) = (double)(i - startx);
			y.at<double>(i, j) = (double)(j - starty);
		}
	}

	Mat nHeight = Mat::zeros(nn, mm, CV_64FC1);

	// 角膜计算区域
	vector<double> Rx1;
	vector<double> Ry1;
	vector<double> Rz1;
	// 确定绘图区域
	for (int i = 0; i < nn; i++)
	{
		for (int j = 0; j < mm; j++)
		{
			double x1 = x.at<double>(i, j);
			double y1 = y.at<double>(i, j);
			double R1 = sqrt(x1 * x1 + y1 * y1); 
			//double z1 = pic.at<double>(i, j) * (depth / nImgLength);  // 横纵比保持一致，并且不还原原图比例尺
			double z1 = pic.at<double>(i, j);  // 前面已经保持横纵比一致了
			if ((R1 <= Rcornea) && (z1 != 0))  // 在半径内，并且值不为0
			{
				//Rx1.push_back(x1 * (width / nImgLength));
				//Ry1.push_back(y1 * (width / nImgLength));   
				Rx1.push_back(x1);   // 横纵比保持一致，并且不还原原图比例尺
				Ry1.push_back(y1);   // 横纵比保持一致，并且不还原原图比例尺
				Rz1.push_back(z1);
			}
		}
	}
	vector<double> center;
	double R_cornea = fitspherebYTandReturn(Rx1, Ry1, Rz1, center);
	double xcenter = center[0];
	double ycenter = center[1];
	double zcenter = center[2];
	//// 可视化查看拟合出的球
	//Mat cornea_fit_sphere = cal_z(xcenter, ycenter, zcenter, R_cornea, dimx);
	//save_pointcloud_aspcd(cornea_fit_sphere, "debug_img/" + fname + "_ball.pcd");

	// 还原真实尺度
	xcenter *= (width / nImgLength), ycenter *= (width / nImgLength), zcenter *= (width / nImgLength), R_cornea *= (width / nImgLength);

	//cout << "R_cornea: " << R_cornea << endl;

	double num1 = *max_element(Rz1.begin(), Rz1.end());
	double num2 = *min_element(Rz1.begin(), Rz1.end());

	int addSize = 64;
	int plotSize = (Rcornea + addSize) * 2 + 1;
	//cout << "plotSize: " << plotSize << endl;
	Mat map1 = Mat::zeros(plotSize, plotSize, CV_64FC1);
	Mat map2 = Mat::zeros(plotSize, plotSize, CV_64FC1);
	//Mat map3 = Mat::zeros(plotSize, plotSize, CV_64FC1);

	// map1用来计算4个方位的曲率半径
	for (int i = 0; i < nn; i++)
	{
		for (int j = 0; j < mm; j++)
		{
			double x1 = x.at<double>(i, j);
			double y1 = y.at<double>(i, j);
			double R1 = sqrt(x1 * x1 + y1 * y1);

			if (R1 <= Rcornea)
			{
				map1.at<double>(x1 + Rcornea + addSize + 0, y1 + Rcornea + addSize + 0) = sqrt(pow((x1 * (width / nImgLength) - xcenter), 2) +
					pow((y1 * (width / nImgLength) - ycenter), 2) +
					//pow((pic.at<double>(i, j) * (depth / nImgLength) - zcenter), 2));
					pow((pic.at<double>(i, j) * (width / nImgLength) - zcenter), 2));  // 前面已经保持横纵比一致了，只需要计算width就行
			}
		}
	}
	// map2用来展示zernike拟合后的曲率半径图
	for (int i = 0; i < nn; i++)
	{
		for (int j = 0; j < mm; j++)
		{
			double x1 = x.at<double>(i, j);
			double y1 = y.at<double>(i, j);
			double R1 = sqrt(x1 * x1 + y1 * y1);

			if (R1 <= Rcornea)
			{
				map2.at<double>(x1 + Rcornea + addSize + 0, y1 + Rcornea + addSize + 0) = sqrt(pow((x1 * (width / nImgLength) - xcenter), 2) +
					pow((y1 * (width / nImgLength) - ycenter), 2) +
					//pow((pic.at<double>(i, j) * (depth / nImgLength) - zcenter), 2));
					pow((zernike_map[i][j] * (width / nImgLength) - zcenter), 2));  // 前面已经保持横纵比一致了，只需要计算width就行
			}
		}
	}


	//
	corneaOutput out;
	out.r_0_0 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize + 0);
	//
	double realR = 1;
	int outR = floor(realR / (width / (double)nImgLength));
	out.r_10_0 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize - outR);
	out.r_10_90 = map1.at<double>(Rcornea + addSize + outR, Rcornea + addSize + 0);
	out.r_10_180 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize + outR);
	out.r_10_270 = map1.at<double>(Rcornea + addSize - outR, Rcornea + addSize + 0);
	//
	realR = 1.4;
	outR = floor(realR / (width / (double)nImgLength));
	out.r_14_0 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize - outR);
	out.r_14_90 = map1.at<double>(Rcornea + addSize + outR, Rcornea + addSize + 0);
	out.r_14_180 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize + outR);
	out.r_14_270 = map1.at<double>(Rcornea + addSize - outR, Rcornea + addSize + 0);
	//
	realR = 2;
	outR = floor(realR / (width / (double)nImgLength));
	out.r_20_0 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize - outR);
	out.r_20_90 = map1.at<double>(Rcornea + addSize + outR, Rcornea + addSize + 0);
	out.r_20_180 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize + outR);
	out.r_20_270 = map1.at<double>(Rcornea + addSize - outR, Rcornea + addSize + 0);
	//
	realR = 2.5;
	outR = floor(realR / (width / (double)nImgLength));
	out.r_25_0 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize - outR);
	out.r_25_90 = map1.at<double>(Rcornea + addSize + outR, Rcornea + addSize + 0);
	out.r_25_180 = map1.at<double>(Rcornea + addSize + 0, Rcornea + addSize + outR);
	out.r_25_270 = map1.at<double>(Rcornea + addSize - outR, Rcornea + addSize + 0);


	cout << "-----------------------------------------------------" << endl;
	cout << "cornea result: " << endl;
	cout << "r_0_0:     " << out.r_0_0 << endl;
	//
	cout << "1mm" << endl;
	cout << "r_10_0:    " << out.r_10_0 << endl;
	cout << "r_10_90:   " << out.r_10_90 << endl;
	cout << "r_10_180:  " << out.r_10_180 << endl;
	cout << "r_10_270:  " << out.r_10_270 << endl;
	//
	cout << "1.4mm" << endl;
	cout << "r_14_0:    " << out.r_14_0 << endl;
	cout << "r_14_90:   " << out.r_14_90 << endl;
	cout << "r_14_180:  " << out.r_14_180 << endl;
	cout << "r_14_270:  " << out.r_14_270 << endl;
	//
	cout << "2mm" << endl;
	cout << "r_20_0:    " << out.r_20_0 << endl;
	cout << "r_20_90:   " << out.r_20_90 << endl;
	cout << "r_20_180:  " << out.r_20_180 << endl;
	cout << "r_20_270:  " << out.r_20_270 << endl;
	//
	cout << "2.5mm" << endl;
	cout << "r_25_0:    " << out.r_25_0 << endl;
	cout << "r_25_90:   " << out.r_25_90 << endl;
	cout << "r_25_180:  " << out.r_25_180 << endl;
	cout << "r_25_270:  " << out.r_25_270 << endl;



	cout << "-----------------------------------------------------" << endl;

	if (~if3D)
	{
		ofstream dataFile;
		dataFile.open(path + "//" + "RcorneaCal_" + fname + ".txt", ofstream::app);
		fstream file(path + "//" + "RcorneaCal_" + fname + ".txt", ios::out);
		dataFile << "cornea result: " << endl;
		dataFile << "r_0_0:     " << out.r_0_0 << endl;
		//
		//cout << "1mm" << endl;
		dataFile << "r_10_0:    " << out.r_10_0 << endl;
		dataFile << "r_10_90:   " << out.r_10_90 << endl;
		dataFile << "r_10_180:  " << out.r_10_180 << endl;
		dataFile << "r_10_270:  " << out.r_10_270 << endl;
		//
		//cout << "1.4mm" << endl;
		dataFile << "r_14_0:    " << out.r_14_0 << endl;
		dataFile << "r_14_90:   " << out.r_14_90 << endl;
		dataFile << "r_14_180:  " << out.r_14_180 << endl;
		dataFile << "r_14_270:  " << out.r_14_270 << endl;
		//
		//cout << "2mm" << endl;
		dataFile << "r_20_0:    " << out.r_20_0 << endl;
		dataFile << "r_20_90:   " << out.r_20_90 << endl;
		dataFile << "r_20_180:  " << out.r_20_180 << endl;
		dataFile << "r_20_270:  " << out.r_20_270 << endl;
		//
		//cout << "2.5mm" << endl;
		dataFile << "r_25_0:    " << out.r_25_0 << endl;
		dataFile << "r_25_90:   " << out.r_25_90 << endl;
		dataFile << "r_25_180:  " << out.r_25_180 << endl;
		dataFile << "r_25_270:  " << out.r_25_270 << endl;
		dataFile.close();

	}


	/*default_random_engine e;
	uniform_real_distribution<double> u(0.1, 0.8);*/

	/*num1 = *max_element(Rz2.begin(), Rz2.end());
	num2 = *min_element(Rz2.begin(), Rz2.end());*/

	double minv, maxv;
	Point pt_min, pt_max;
	minMaxLoc(map2, &minv, &maxv, &pt_min, &pt_max);
	num1 = maxv;
	num2 = minv;

	if (if3D) // 不走这步
	{
		ofstream dataFile;
		dataFile.open(path + "//" + "Rcornea.txt", ofstream::app);
		fstream file(path + "//" + "Rcornea.txt", ios::out);
		int numSel = 0;
		// 确定绘图区域map2
		for (int i = 0; i < nn; i++)
		{
			for (int j = 0; j < mm; j++)
			{
				double x1 = x.at<double>(i, j);
				double y1 = y.at<double>(i, j);
				double R1 = sqrt(x1 * x1 + y1 * y1);

				if (R1 <= Rcornea)
				{


					//cout << u(e) << endl;
					double z1 = map2.at<double>(x1 + Rcornea + addSize + 0, y1 + Rcornea + addSize + 0);
					/*Rx2.push_back(x1* (width / nImgLength));
					Ry2.push_back(y1* (width / nImgLength));
					Rz2.push_back(map1.at<double>(x1 + Rdw + addSize + 0, y1 + Rdw + addSize + 0));*/

					numSel = numSel + 1;
					if (numSel % 10 == 0)
					{
						vector<int> color;
						colorToParula(z1, num1, num2, color);
						/*Vector3D* dot = new Vector3D(x1, y1, z1, (uint8_t)color[0], (uint8_t)color[1], (uint8_t)color[2]);*/
						/*dots.push_back(dot);*/
						dataFile << x1 << ' ' << y1 << ' ' << z1 << ' ' << color[0] << ' ' << color[1] << ' ' << color[2] << ' ' << endl;
						//pts->InsertNextPoint(x1, y1, z1);
					}
					//map3.at<double>(x1 + Rdw + addSize + 0, y1 + Rdw + addSize + 0) = abs(map2.at<double>(x1 + Rdw + addSize + 0, y1 + Rdw + addSize + 0) - map1.at<double>(x1 + Rdw + addSize + 0, y1 + Rdw + addSize + 0));
				}

			}
		}
		dataFile.close();

	}


	if (R_cornea > 2.0)
	{
		//
		string name1 = "corneaMap_zernike_" + fname;
		//
		Mat display1;
		Mat display2;
		//Mat mask = Mat::zeros(plotSize, plotSize, CV_8UC1);
		//ColorMapNew picNew;
		//map1.convertTo(display1, CV_8UC1, 255.0 / (num1 - num2), 0);
		map2.convertTo(display1, CV_8UC1, 255.0 / (num1 - num2), 0);
		applyColorMapYT(display1, display2, cv::COLORMAP_PARULA);

		// 写数值
		//cornea_draw_values(display2, map1, width, nImgLength, Rcornea, addSize);
		cornea_draw_values(display2, map2, Rcornea, addSize);
		Mat bar1 = colorBar(plotSize, 0, num1 - num2);
		Mat displayAll;

		if (colorbarPos == 1)
		{
			hconcat(bar1, display2, displayAll);
		}
		else
		{
			hconcat(display2, bar1, displayAll);
		}

		//imshow(name, display2);
		imwrite(path + "//" + name1 + ".jpg", displayAll);

		return R_cornea;
	}
	else
	{
		R_cornea = -1;
		cout << "The radius is wrong or out of range." << endl;
		return R_cornea;
	}





}



void cal_curvature_map(int nPicNum, int range_window_s, Mat nmLayerx, Mat nmLayery, Mat nmHeight_top,
	double width, double depth, int nImgLength, int colorbarPos, string path, string fname,
	double R1, double R2, double R3, bool if3D, double& R_sclera, double& R_cornea) {
	//
	double* mLayerx = new double[nPicNum * range_window_s];
	double* mLayery = new double[nPicNum * range_window_s];
	double* mThickCO = new double[nPicNum * range_window_s];

	int number;
	//	// openMP
	//	omp_set_num_threads(coreNumber);
	//#pragma omp parallel for
	for (int i = 0; i < nPicNum; i++)
	{
		for (int j = 0; j < range_window_s; j++)
		{
			number = j + (i - 0) * range_window_s;
			mLayerx[number] = nmLayerx.at<double>(i, j);
			mLayery[number] = nmLayery.at<double>(i, j);
			//mThickCO[number] = nmHeight_top.at<double>(i, j);  
			mThickCO[number] = nmHeight_top.at<double>(i, j) * depth / width;  // 保持横纵比一致
			/*number = number + 1;*/
		}
	}
	
	double valueMax = (*max_element(mThickCO, mThickCO + nPicNum * range_window_s));  // 保证里面的值为正数，有的时候负数会导致zernike拟合出现边缘翘起的情况
	for (int i = 0; i < nPicNum * range_window_s; i++)
	{
		mThickCO[i] = valueMax - mThickCO[i];
	}
	
	int dim = range_window_s+1;
	double** map = new  double* [dim];
	for (int i = 0; i < dim; i++)
	{
		map[i] = new double[dim]();
	}
	double** zernike_map = new  double* [dim];
	for (int i = 0; i < dim; i++)
	{
		zernike_map[i] = new double[dim]();
	}

	twelve_line(mLayerx, mLayery, mThickCO, number, dim, map);
	////// 保存pcd三维图
	Mat cornea_map_mat = array2mat(map, dim);
	save_pointcloud_aspcd(cornea_map_mat, "debug_img/" + fname + "_cornea_twelve_line.pcd");

	cout << "fUNCTION griddata C++version test: " << endl;
	griddataFun(mLayerx, mLayery, mThickCO, number, dim, zernike_map);  // griddata
	////// 保存pcd三维图
	//Mat cornea_map_mat = array2mat(map, dim);
	//save_pointcloud_aspcd(cornea_map_mat, "debug_img/" + fname + "_cornea_griddata2.pcd");

	//Mat map1;
	double Rcornea = floor(R1 / (width / (double)nImgLength));
	mapZernikeCUDA(zernike_map, (int)Rcornea, dim);
	////// 保存pcd三维图
	//cornea_map_mat = array2mat(map, dim);
	//save_pointcloud_aspcd(cornea_map_mat, "debug_img/" + fname + "_cornea_griddata_Zernike2.pcd");
	R_cornea = RcalCornea(map, zernike_map, dim, dim, width, depth, nImgLength, 3.75, colorbarPos, path, fname, R1, if3D);

	delete[] mLayerx;
	delete[] mLayery;
	delete[] mThickCO;
	for (int i = 0; i < dim; i++)
	{
		delete[] zernike_map[i];
	}
	delete[] zernike_map;
	for (int i = 0; i < dim; i++)
	{
		delete[] map[i];
	}
	delete[] map;
}





void ScleraMap::mapCon(string pathFile[], string filePathOut[], int nPicNum, string path, double& R_sclera, double& R_cornea)
{
	// 基本参数定义
	double R1 = m_config.Rcornea;
	double R2 = m_config.Rup;
	double R3 = m_config.Rdw;
	bool if3D = m_config.if3D;
	int nSetpAngle = 180 / nPicNum;
	int nImgDeep = m_config.nImgDeep;  //
	int	nImgLength = m_config.nImgLength;  //
	double xCorr = m_config.xCorr;
	double yCorr = m_config.yCorr;
	double	width = m_config.width * xCorr; //
	double	depth = m_config.depth * yCorr;  //
	double	index = m_config.index;  //
	int	eye = m_config.eye; //eye = 1, OS; eye = 2, OD
	int	range_window_s = (floor(R1 / (width / (double)nImgLength)) * 2) + 200; // R == = 850;
	int half_size = floor(range_window_s / 2);
	//cout << half_size << endl;
	int colorbarPos = m_config.colorbarPos;
	int cutSize = m_config.cutSize;
	int fineHeight_startidx = m_config.fineHeight_startidx;  // 分割出的上界线往下平移多少像素
	int fineHeight = m_config.fineHeight;  // 分割区域的范围h

	//
	Mat nmLayerx = Mat::zeros(nPicNum, range_window_s, CV_64FC1);
	Mat nmLayery = Mat::zeros(nPicNum, range_window_s, CV_64FC1);
	Mat nmHeight_top = Mat::zeros(nPicNum, range_window_s, CV_64FC1);
	Mat nmHeight_bottom = Mat::zeros(nPicNum, range_window_s, CV_64FC1);

	//
	Mat mHeight_top = Mat::zeros(nPicNum, nImgLength, CV_64FC1);
	Mat mHeight_bottom = Mat::zeros(nPicNum, nImgLength, CV_64FC1);
	Mat mLineSame = Mat::ones(1, nImgLength, CV_64FC1);
	Mat mLineInc_top = Mat::zeros(1, nImgLength, CV_64FC1);
	Mat mLineInc_bottom = Mat::zeros(1, nImgLength, CV_64FC1);
	Mat mLineInc_top_all = Mat::zeros(1, nImgLength, CV_64FC1);
	Mat mLineInc_bottom_all = Mat::zeros(1, nImgLength, CV_64FC1);
	Mat Center_top_X_R = Mat::zeros(1, nPicNum, CV_64FC1);
	Mat Center_bottom_X_R = Mat::zeros(1, nPicNum, CV_64FC1);

	//
	for (int i = 0; i < nImgLength; i++)
	{
		mLineInc_top.at<double>(0, i) = (double)i;
	}


	for (int i = 0; i < nImgLength; i++)
	{
		mLineInc_bottom.at<double>(0, i) = (double)i;
	}

	//
	int coreGet = omp_get_num_procs();
	cout << "totalnumber of core: " << coreGet << endl;
	int coreNumber = 1;
	if (coreGet >= 12)
	{
		coreNumber = 12;
	}
	else if (coreGet >= 6)
	{
		coreNumber = 6;
	}
	else if (coreGet >= 4)
	{
		coreNumber = 4;
	}
	else if (coreGet >= 2)
	{
		coreNumber = 2;
	}
	omp_set_num_threads(coreNumber);
#pragma omp parallel for
	for (int i = 0; i < nPicNum; i++)
	{
		string filename1 = pathFile[i];
		/*cout << filename << endl;*/
		Mat pic = imread(filename1, IMREAD_GRAYSCALE);

		if (pic.empty())
		{  // 校验是否正常打开待操作图像!
			cout << "can't open the image!!!!!!!" << endl;
		}
		double* label_top = new double[nImgLength];
		double* label_bottom = new double[nImgLength];

		string filename2 = filePathOut[i];

		corneaSegLine(filename2, pic, label_top, label_bottom, cutSize, Center_top_X_R.at<double>(0, i), Center_bottom_X_R.at<double>(0, i),
			 fineHeight_startidx, fineHeight);
		for (int j = 0; j < nImgLength; j++)
		{
			mHeight_top.at<double>(i, j) = label_top[j];
			mHeight_bottom.at<double>(i, j) = label_bottom[j];
		}
		//cout << label_top[1024];


		delete[] label_top;
		delete[] label_bottom;

		double nAngle = (double)(nSetpAngle * (i - 0));
		double zeros_top = Center_top_X_R.at<double>(0, i);
		double zeros_bottom = Center_bottom_X_R.at<double>(0, i);
		Mat mx = Mat::zeros(1, range_window_s, CV_64FC1);
		Mat my = Mat::zeros(1, range_window_s, CV_64FC1);
		double theta;
		if (eye == 1)
		{
			theta = -nAngle / 180 * M_PI;
		}
		else
		{
			theta = -(180 - nAngle) / 180 * M_PI;
		}
		for (int j = 0; j < range_window_s; j++)
		{
			double ro = (double)(-half_size + j);
			mx.at<double>(0, j) = ro * cos(theta);
			my.at<double>(0, j) = ro * sin(theta);
			nmLayerx.at<double>(i, j) = mx.at<double>(0, j);
			nmLayery.at<double>(i, j) = my.at<double>(0, j);
			nmHeight_top.at<double>(i, j) = mHeight_top.at<double>(i, zeros_top - half_size + j);
			nmHeight_bottom.at<double>(i, j) = mHeight_bottom.at<double>(i, zeros_bottom - half_size + j);
		}
	}

	cal_curvature_map(nPicNum, range_window_s, nmLayerx, nmLayery, nmHeight_top,
		width, depth, nImgLength, colorbarPos, path, "top", R1, R2, R3, if3D, R_sclera, R_cornea);

	cal_curvature_map(nPicNum, range_window_s, nmLayerx, nmLayery, nmHeight_bottom,
		width, depth, nImgLength, colorbarPos, path, "bottom", R1, R2, R3, if3D, R_sclera, R_cornea);
}


// 初始化标定数据
void ScleraMap::setConfig(const struct Config& config)
{
	m_config = config;
}