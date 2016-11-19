#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <algorithm>
#include <iterator>
#include <set>

#define PI 3.14


class Distribution
{
public:
	Distribution();
	~Distribution();
public:
	void ditributionGrayScale(cv::Mat &input_img, float R1, float &R2);
	void drawDistribution(std::vector<float> &arrays,int &X_Scale,int &Y_Scale);
	cv::Mat sobelMethod(cv::Mat &input_img);
	cv::Mat scharrMethod(cv::Mat &input_img);

	cv::Mat binaryThreshold(cv::Mat &input_img,int &value);


	cv::Mat polarCoordinate(cv::Mat &input_img, float &R1, float &R2);
};

