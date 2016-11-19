#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2\features2d\features2d.hpp"
#include <fstream>

#include "Distribution.h"
#include "LineDistribution.h"
#include "MaximaLocalMethod.h"
#include "Timer.h"
#include "FrameTimer.h"

cv::Mat src, src2, src3; 
cv::Mat src_gray;
int thresh = 53;
int max_thresh = 200;
cv::RNG rng(12345);
Distribution dist_class;

/// Function header
void thresh_callback(int, void*);
int main(int argc, char** argv){


	std::wofstream logFile(L"logfile.txt");
	FrameTimer ft;
	MaximaLocalMethod local_maxima;
	cv::Mat input_Grayimg = cv::imread("image_roi1.bmp", 0);
    cv::Mat out_Grayimg;

	
	
	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 50;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.5;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
	// Set up the detector with default parameters.
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	// Detect blobs.
	std::vector<cv::KeyPoint> keypoints;

	int k = 0;
	cv::Mat im_with_keypoints;
	while (k!=1000)
	{
		ft.StartFrame();
		local_maxima.InvertGray255SSE(input_Grayimg, out_Grayimg);
		/*cv::cvtColor(input_Grayimg, out_Grayimg, cv::COLOR_BGR2GRAY);
		detector->detect(out_Grayimg, keypoints);
		drawKeypoints(out_Grayimg, keypoints, im_with_keypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);*/
		ft.StopFrame(logFile);
		cv::imshow("gray", out_Grayimg);
	/*	cv::imshow("keypoint", im_with_keypoints);*/
		cv::waitKey(10);
		k++;
	}
	/*cv::imwrite("invertGray255_key.bmp", im_with_keypoints);*/
	cv::imwrite("invertGray255.bmp", out_Grayimg);
	// Previous work: before 11/19/2016

#pragma region Previous
	
	//src2 = cv::imread("template.bmp", 0);
	///*cv::Mat sobel = dist_class.scharrMethod(src2);
	//cv::imwrite("sobel_temple.bmp",sobel);*/
	//cv::Mat image = cv::imread("template.bmp", 0);

	//cv::Mat dst;
	//cv::linearPolar(image, dst, cv::Point(image.cols / 2, image.rows / 2), image.rows / 2, cv::INTER_CUBIC);
	//cv::imshow("linear", dst);

	//cv::namedWindow("ctrl");
	//int win = 62;
	//int th = 2100;
	//cv::createTrackbar("win", "ctrl", &win, 500);
	//cv::createTrackbar("th", "ctrl", &th, 10000);
	//
	//while (true)
	//{
	//	cv::Mat thresh;
	//	image.copyTo(thresh);
	//	/*cv::medianBlur(src2, thresh, 15);*/
	//	adaptiveThreshold(thresh, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, win * 2 + 1, (th / 1000.));
	//	imshow("thresh", thresh);
	//	cv::imwrite("adaptive.bmp", thresh);
	//	cv::waitKey(10);
	//}

	///*image.copyTo(src);
	//image.copyTo(src_gray);
	//float R1 = (float)src_gray.cols / 2 ;
	//float R2 = (float)src_gray.cols / 2 - 90 ;*/
	///*blur(src_gray, src_gray, cv::Size(5, 5));*/
	///*dist_class.ditributionGrayScale(src_gray, R1, R2);*/
	///*dist_class.polarCoordinate(src_gray, R1, R2);*/

	///*cv::imshow("original", image);*/

	//
	///*while (true)
	//{

	//	cv::waitKey(10);
	//}*/
	//
	////src = cv::imread("image_roi1.bmp", 1);
	////src_gray = cv::imread("gray_scale.bmp", 0);


	//////cv::Mat sobel_img = dist_class.sobelMethod(src);
	//////cv::Mat scharr_img = dist_class.scharrMethod(src);
	//////cv::imwrite("sobel_img.bmp", sobel_img);
	//////cv::imwrite("scahrr_img.bmp", scharr_img);


	///*blur(src_gray, src_gray, cv::Size(5,5));

	///// Create Window
	//char* source_window = "Source";
	//cv::namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	//imshow(source_window, src);

	//cv::createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
	//thresh_callback(0, 0);*/

#pragma endregion ending previsous research
	return 0;
}


void thresh_callback(int, void*)
{
	cv::Mat canny_output,canny_ouyr_2;
	/*Canny(src_gray, canny_output, thresh, (double)thresh * 2, 3);*/
	canny_output = dist_class.binaryThreshold(src_gray, thresh);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	canny_output.copyTo(canny_ouyr_2);
	findContours(canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i< contours.size(); i++)
	{
		if (contours[i].size() < 20) continue;

		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(src2, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
	}

	cv::imshow("gray scale", canny_ouyr_2);
	cv::imshow("contour", src2);
	cv::imwrite("sobel_polar.bmp", canny_ouyr_2);
	cv::waitKey(10);
	
	//float R1 = src_gray.cols / 2;
	//float R2 = src_gray.cols / 2 - 95;
	//float sigma = 9;
	//cv::Point2f center = cv::Point(src_gray.cols / 2, src_gray.rows / 2);

	//LineDistribution line_distribution;

	//cv::Mat canny_output;
	//std::vector<std::vector<cv::Point> > contours;
	//std::vector<cv::Vec4i> hierarchy;
	//

	///// Detect edges using canny
	//Canny(src_gray, canny_output, thresh, (double)thresh * 1.3, 3); cv::Mat detect_line; canny_output.copyTo(detect_line);
	//cv::imshow("canny", canny_output);
	//cv::Mat canny_blur;
	//blur(canny_output, canny_blur, cv::Size(5, 5));
	//int value = 45;
	//canny_blur = dist_class.binaryThreshold(canny_blur, value);
	//cv::imwrite("canny.bmp", canny_output);
	//cv::imwrite("canny_blur.bmp", canny_blur);
	//cv::imshow("canny blur", canny_blur);
	//
	//
	///// Find contours
	//findContours(canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//
	///// Draw contours
	//cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
	//cv::Mat color_img;// = src;
	//src.copyTo(color_img);
	//for (int i = 0; i != contours.size(); i++)
	//{
	//	if (contours[i].size() < 20) continue;
	//	/*if (contours[i].size() > 200) continue;*/
	//	/*float max_length_contour = 0;
	//	for (int j = 0; j != contours[i].size(); j++){
	//	unsigned int idx1 = rand() % contours[i].size();
	//	unsigned int idx2 = rand() % contours[i].size();
	//	if (idx1 == idx2) continue;
	//	float temp_distance = cv::norm(contours[i][idx1] - contours[i][idx2]);
	//	if (temp_distance > max_length_contour){
	//	max_length_contour = temp_distance;
	//	}
	//	}
	//	if (max_length_contour < 20) continue;*/


	//	// checking ria duong truong
	//	bool found_out = line_distribution.delectContourPoints_Out(contours[i], R1, sigma);
	//	if (found_out) continue;

	//	bool found_in = line_distribution.delectContourPoints_In(contours[i], R1, R2, sigma);
	//	if (found_in) continue;
	//	

	//	

	//	cv::Mat pointsf;
	//	cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
	//	cv::RotatedRect temp = cv::fitEllipse(pointsf);
	//	

	//	float radius_temp = cv::norm(temp.center - center);
	//	if (radius_temp < R2 || radius_temp > R1) continue;
	//	
	//
	//	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	// convex Hull
	//	std::vector<int> hull;
	//	std::vector<cv::Vec4i> convecDefect;
	//	cv::convexHull(pointsf, hull, false);
	//	cv::convexityDefects(contours[i], hull, convecDefect);

	//	cv::line(drawing, contours[i][hull[0]], contours[i][hull[hull.size()/2]], color, 1, 8);

	//	double max_norm = cv::norm(contours[i][hull[0]] - contours[i][hull[hull.size() / 2]]);

	//	
	//	/*cv::line(drawing, contours[i][hull[hull.size() / 2]], contours[i][hull[hull.size()-1]], color, 1, 8);*/
	//	char size_text[20];
	//	sprintf_s(size_text, "%.1f", max_norm);
	//	cv::putText(drawing, size_text, temp.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 8);
	//	// drawing size


	//	
	//	if (max_norm < 35) continue;


	//	// draw defect
	//	cv::ellipse(color_img, temp, cv::Scalar(0, 255, 0), 1, 0);
	//	// checking standard diviation
	//	float sigma_gradient = line_distribution.calGradientSigma(contours[i]);
	//	//if (sigma_gradient >= 1.15) continue;
	//	char text[50];
	//	sprintf_s(text, "defected %.1f", sigma_gradient);
	//	//std::string text = std::to_string(max_length_contour) + ":"  + std::to_string(contours[i].size());
	//	if (sigma_gradient < 1){
	//		cv::putText(color_img, text, temp.center, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(233, 0, 255), 1, 8);
	//	}
	//	else{
	//		cv::putText(color_img, text, temp.center, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 1, 8);
	//	}
	//	
	//}

	////// detect line sigment
	////cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);//LSD_REFINE_STD
	////std::vector<cv::Vec4f> lines_std;
	////// Detect the lines
	////ls->detect(detect_line, lines_std);
	////ls->drawSegments(color_img, lines_std);
	///// Show in a window
	//cv::namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//cv::imshow("Contours", drawing);
	//cv::imshow("defect", color_img);
	//cv::imwrite("defect.bmp", color_img);
	//cv::imwrite("contour.bmp", drawing);
}