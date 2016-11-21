#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <assert.h>

class MaximaLocalMethod
{
public:
	MaximaLocalMethod();
	~MaximaLocalMethod();

	void InvertGray255(cv::Mat &input_img, cv::Mat &output_img);
	void InvertGray255SSE(cv::Mat &input_img, cv::Mat &output_img);
	void InvertGray255AVX(cv::Mat &input_img, cv::Mat &output_img);
	void SobelMultipleLevelAdding(cv::Mat &input_img,cv::Mat &output_ing, int &number_times);
};

