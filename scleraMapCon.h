#ifndef CORNEAMAPCON_H
#define CORNEAMAPCON_H
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

double weight(double ga, double gb);

class ScleraMap
{
	/*public:
		enum errorCode { POSITION, SIGNAL, NONE };
		errorCode error*/;

public:
	struct Config {
		int nImgDeep = 2048;
		int	nImgLength = 2048;
		double width = 19.56;
		double depth = 7.10;
		double index = 1.38;
		
		int eye = 1;
		int mode = 1;
		int window = 100;
		//int	nImgLength_window = nImgLength - 2 * window + 1;		
		int colorbarPos = 1; // 1的时候在右边，0在左边；
		int cutSize = 450;
		double xCorr = 1.032;
		double yCorr = 0.91;
		double Rcornea = 3.75;
		double Rup = 2;
		double Rdw = 4;
		bool if3D = false;
		int fineHeight_startidx = 100;
		int fineHeight = 101;
	};

public:
	//double r_radius_sclera;
	double r_sclera;
	double r_cornea;
	void mapCon(string pathFile[], string filePathOut[], int nPicNum, string path, double & R_sclera, double & R_cornea);
	void setConfig(const struct Config& config);
	

private:
	Config m_config;
};

class corneaOutput
{
public:
	double r_0_0;
	//
	double r_10_0;
	double r_10_90;
	double r_10_180;
	double r_10_270;
	//
	double r_14_0;
	double r_14_90;
	double r_14_180;
	double r_14_270;
	//
	double r_20_0;
	double r_20_90;
	double r_20_180;
	double r_20_270;
	//
	double r_25_0;
	double r_25_90;
	double r_25_180;
	double r_25_270;
	//
	double r_31_0;
	double r_31_90;
	double r_31_180;
	double r_31_270;
	//
	double r_35_0;
	double r_35_90;
	double r_35_180;
	double r_35_270;
};

class scleraOutput
{
public:
	//
	double r_45_0;
	double r_45_90;
	double r_45_180;
	double r_45_270;
	//
	double r_50_0;
	double r_50_90;
	double r_50_180;
	double r_50_270;
	//
	double r_65_0;
	double r_65_90;
	double r_65_180;
	double r_65_270;
	//
	double r_70_0;
	double r_70_90;
	double r_70_180;
	double r_70_270;


};

//class output
//{
//public:
//	double r_cornea;
//	double r_sclera;
//};
#endif