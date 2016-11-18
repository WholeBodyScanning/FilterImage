#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>

class LineDistribution
{
public:
	LineDistribution();
	~LineDistribution();
public:
	void drawCilce(cv::Mat &input_img);
	void drawLineDistribution(cv::Mat &input_img,  int &number_row);
	void delectOutlierPixel(cv::Mat &input_img,int &R1,int &R2);
	bool delectContourPoints_Out(std::vector<cv::Point> &points, float &R1,float &sigma);
	bool delectContourPoints_In(std::vector<cv::Point> &points, float &R1, float &R2, float &sigma);
	float calGradientSigma(std::vector<cv::Point> &points);
};

