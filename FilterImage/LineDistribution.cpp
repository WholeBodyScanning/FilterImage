#include "LineDistribution.h"


LineDistribution::LineDistribution()
{
}


LineDistribution::~LineDistribution()
{
}

void LineDistribution::drawCilce(cv::Mat &input_img){
	cv::Mat image_gray;
	int R1 = input_img.cols / 2;
	int R2 = input_img.cols / 2 - 95;
	cvtColor(input_img, image_gray, CV_BGR2GRAY);
	delectOutlierPixel(image_gray, R1, R2);

	circle(input_img, cv::Point(input_img.cols / 2, input_img.rows / 2), input_img.cols / 2, cv::Scalar(0, 0, 255), 1, 8, 0);
	circle(input_img, cv::Point(input_img.cols / 2, input_img.rows / 2), input_img.cols / 2 - 95, cv::Scalar(0, 0, 255), 1, 8, 0);
	//cv::ellipse(input_img, cv::Point(input_img.cols, input_img.rows), cv::Size(200, 100), -30, 180, 270, cv::Scalar(0, 0, 255));
	//std::vector<cv::Point> point_pts;
	//cv::ellipse2Poly(cv::Point(input_img.cols/2, input_img.rows/2), cv::Size(100, 50), -30, 0, 360, 1, point_pts);
	//for (int i = 0; i != point_pts.size(); i++){
	//	circle(input_img, point_pts[i], 1, cv::Scalar(0, 0, 255), 1, 8, 0);
	//	/*std::cout << point_pts[i] << std::endl;*/
	//}
	//std::cout << "size: " << point_pts.size() << std::endl;
	
	
	cv::imshow("gray", image_gray);
	cv::imshow("cirle", input_img);
	cv::waitKey(10);
}

/*
  input gray iamge
*/
void LineDistribution::delectOutlierPixel(cv::Mat &input_img,int &R1,int &R2){

	cv::Point center = cv::Point(input_img.cols / 2, input_img.rows / 2);

	for (int i = 0; i != input_img.rows; i++){
		uchar *Mi = input_img.ptr<uchar>(i);
		for (int j = 0; j != input_img.cols; j++){
			cv::Point point_pt = cv::Point(j, i);
			int temp_dis = cv::norm(point_pt - center);

			if (temp_dis <= R1 && temp_dis >= R2) Mi[j] = Mi[j];
			else Mi[j] = 0;

		}
	}
}

bool LineDistribution::delectContourPoints_Out(std::vector<cv::Point> &points, float &R1, float &sigma){

	cv::Point center = cv::Point(R1,R1);
	float counter_out = 0;
	float max_threshold_out = 30*points.size()/100;
	std::vector<cv::Point>::iterator point_pts = points.begin();

	for (; point_pts != points.end(); point_pts++){
		float temp_radius = cv::norm(*point_pts - center);
		float out_radius = cv::abs(temp_radius - R1);
		
		if (out_radius < sigma ){
			counter_out++;
			
		}
		
	}

	if (counter_out >= max_threshold_out) return 1;
	else return 0;
}

bool LineDistribution::delectContourPoints_In(std::vector<cv::Point> &points,float &R1, float &R2, float &sigma){
	cv::Point center = cv::Point(R1, R1);
	float counter_in = 0;
	float max_threshold_out = 30 * points.size() / 100;
	std::vector<cv::Point>::iterator point_pts = points.begin();

	for (; point_pts != points.end(); point_pts++){
		float temp_radius = cv::norm(*point_pts - center);
		float in_radius = cv::abs(temp_radius - R2);

		if (in_radius < sigma){
			counter_in++;

		}

	}

	if (counter_in >= max_threshold_out) return 1;
	else return 0;
}

void LineDistribution::drawLineDistribution(cv::Mat &input_img,  int &number_row){
	cv::Mat image_gray,image_color,image_color_show;
	input_img.copyTo(image_color_show);
	image_color = input_img.clone();
	cvtColor(input_img, image_gray, CV_BGR2GRAY);

	uchar *Mi = image_gray.ptr<uchar>(number_row);

	for (int j = 0; j != image_gray.cols; j++){
		int value = Mi[j];

		cv::line(image_color, cv::Point(j, value), cv::Point(j,input_img.rows ), cv::Scalar(255, 255, 0), 1, 8);
	}

	cv::line(image_color_show, cv::Point(0,number_row),cv::Point(input_img.cols,number_row), cv::Scalar(255, 255, 0), 1, 8);
	cv::imshow("original", image_color_show);
	cv::imshow("histogram", image_color);
	cv::waitKey(10);

	cv::Mat grad_x, abs_grad_x;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_REPLICATE);
	convertScaleAbs(grad_x, abs_grad_x);
	cv::imshow("gradient x", abs_grad_x);
	cv::waitKey(10);
}

float LineDistribution::calGradientSigma(std::vector<cv::Point> &points){
	
	float N = 0;
	
	std::vector<float> sum_gradient;
	sum_gradient.reserve(N);

	float total_average = 0;

	for (int i = 0; i != points.size(); i++){

		unsigned int idx1 = rand() % points.size();
		unsigned int idx2 = rand() % points.size();
		if (idx1 == idx2) continue;

		float dominator = (points[idx1].x - points[idx2].x);
		if (dominator == 0){
			sum_gradient.push_back(0);

			total_average += 0;
		}
		else{
			float fraction = points[idx1].y - points[idx2].y;

			float slope = fraction / dominator;
			sum_gradient.push_back(slope);
			total_average += slope;
		}

		N++;
	}
	// mean
	float mean = total_average / N;

	float sigma_square = 0;
	std::vector<float>::iterator sum_pts = sum_gradient.begin();
	for (; sum_pts != sum_gradient.end(); sum_pts++){

		float temp = (*sum_pts - mean);
		sigma_square += temp*temp;
	}

	return std::sqrt(sigma_square / N);

}
