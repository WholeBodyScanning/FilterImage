#include "Distribution.h"


Distribution::Distribution()
{
}


Distribution::~Distribution()
{
}

void Distribution::drawDistribution(std::vector<float> &arrays, int &X_Scale, int &Y_Scale){
	cv::Mat image_color(cv::Size(arrays.size() * X_Scale, arrays.size() * Y_Scale + 30 * Y_Scale), CV_8UC3);

	int index = 0;
	for (int i = 0; i != arrays.size(); i++){
		int hist = arrays[i];
		if (i % 10 == 0){
			std::string text = std::to_string(i % 100);
			cv::putText(image_color, text, cv::Point(index, 30), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, 8);
		}
		else{
			std::string text = std::to_string(i % 10);
			cv::putText(image_color, text, cv::Point(index, 30), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1, 8);
		}
		
		/*td::cout << "hist " << i << " : " << hist << std::endl;*/
		for (int j = 0; j != X_Scale; j++){
			cv::line
				(image_color
				, cv::Point(index, 255*Y_Scale - hist*Y_Scale + 30*Y_Scale), cv::Point(index, image_color.rows)
				, cv::Scalar(10, 120, 255)
				);
			index++;
		}
	}

	/*cv::imshow("histogram", image_color);
	cv::imwrite("histogram_roi1.bmp", image_color);*/
	cv::waitKey(10);
}

void Distribution::ditributionGrayScale(cv::Mat &input_img, float R1, float &R2){
	cv::Point center = cv::Point(input_img.cols / 2, input_img.rows / 2);
	std::vector<float> arrays_pts;
	float initial_value = 0;
	const int hist_height = 256;
	float total_counter = 0;

	arrays_pts.assign(256, initial_value);

	for (int i = 0; i != input_img.rows; i++){
		uchar *Mi = input_img.ptr<uchar>(i);
		for (int j = 0; j != input_img.cols; j++){
			cv::Point point_pt = cv::Point(j, i);
			int temp_dis = cv::norm(point_pt - center);

			if (temp_dis <= R1 && temp_dis >= R2){
				/*Mi[j] = Mi[j];*/
				int temp_value = Mi[j];
				if(temp_value == 0) continue;
				arrays_pts[temp_value] += 1;
				total_counter += 1;
				// check near R1 and r2

				int temp_near = temp_dis - R2;
				

				if (temp_near < 5) continue;
				

				// threshold
				if (temp_value > 160) continue;
				int gray_scale = cv::abs(temp_value - 160);
				if (gray_scale > 10) Mi[j] = 100;
			}
			else Mi[j] = 0;

		}
	}

	// find max value
	float max_value = 0;
	for (auto array_value : arrays_pts){
		float temp_value = array_value;
		if (temp_value > max_value){
			max_value = temp_value;
		}
	}

	// visualize each bin

	std::vector<float>::iterator arrays_value = arrays_pts.begin();

	for (; arrays_value != arrays_pts.end();arrays_value++){
		float temp_value = *arrays_value * hist_height / max_value;

		*arrays_value = temp_value;

	}

	int X_Scale = 9;
	int Y_Scale = 2;

	drawDistribution(arrays_pts, X_Scale, Y_Scale);

	
	/*cv::imshow("gray scale", input_img);*/
	cv::imwrite("gray_scale.bmp", input_img);
}



cv::Mat Distribution::sobelMethod(cv::Mat &input_img){
	cv::Mat src, src_gray;
	cv::Mat grad;
	
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	input_img.copyTo(src);

	GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
	/// Generate grad_x and grad_y
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	/*Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );*/
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_REPLICATE);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	/*Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );*/
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_REPLICATE);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;
}

cv::Mat Distribution::scharrMethod(cv::Mat &input_img){
	cv::Mat src, src_gray;
	cv::Mat grad;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	input_img.copyTo(src);

	GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
	/// Generate grad_x and grad_y
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;
}


cv::Mat Distribution::binaryThreshold(cv::Mat &input_img,int &value){
	cv::Point center = cv::Point(input_img.cols / 2, input_img.rows / 2);

	float R1 = (float)input_img.cols / 2-7;
	float R2 = (float)input_img.cols / 2 - 120;

	cv::Mat gray_img;
	input_img.copyTo(gray_img);

	for (int i = 0; i != gray_img.rows; i++){
		uchar *Mi = gray_img.ptr<uchar>(i);

		for (int j = 0; j != gray_img.cols; j++){

			cv::Point point_pt = cv::Point(j, i);
			int temp_dis = cv::norm(point_pt - center);

			if (temp_dis <= R1 && temp_dis >= R2){

				int temp_near = temp_dis - R2;

				if (temp_near < 5) continue;

				if (Mi[j] >= value) Mi[j] = 255;
				else Mi[j] = 0;
			}
			else{
				Mi[j] = 0;
			}

			
		}
	}


	return gray_img;
}
cv::Mat Distribution::polarCoordinate(cv::Mat &input_img, float &R1, float &R2){
	cv::Point center = cv::Point(input_img.cols / 2, input_img.rows / 2);
	float inverse_y = input_img.rows;
	std::vector<std::vector<cv::Point>> array_angle;
	array_angle.resize(361);
	std::vector<float> arrays_pts;
	arrays_pts.assign(361, 0);

	// find element for array
	for (int i = 0; i != input_img.rows; i++){
		uchar *Mi = input_img.ptr<uchar>(i);
		for (int j = 0; j != input_img.cols; j++){

			cv::Point point_pt = cv::Point(j, i);
			int temp_dis = cv::norm(point_pt - center);

			if (Mi[j] == 255){


				float temp_y = i - center.y;
				float temp_x = j - center.x;
				
				// float theta
				float theta = atan2f(temp_y, temp_x);
				theta *= 180 / PI;
				int theta_value = theta;

				if (i == 509 && j == 104){
					std::cout << ">0: " << theta_value << std::endl;
				}

				if (theta_value < 0){
					/*std::cout << ">0" << std::endl;*/

					
					theta_value = 360 + theta_value;
					if (i == 509 && j == 104){
						std::cout << ">0: " << theta_value<< std::endl;
					}
				}
				arrays_pts[theta_value] += 1;
				// 
				array_angle[theta_value].push_back(cv::Point(j,i));
			}
		}
	}
	// find max value
	float max_value = 0;
	for (auto array_value : arrays_pts){
		float temp_value = array_value;
		if (temp_value > max_value){
			max_value = temp_value;
		}
	}

	std::vector<float>::iterator arrays_value = arrays_pts.begin();
	const int hist_height = R1-R2;
	for (; arrays_value != arrays_pts.end(); arrays_value++){
		float temp_value = *arrays_value * hist_height / max_value;

		*arrays_value = temp_value;

	}

	int X_Scale = 9;
	int Y_Scale = 2;

	drawDistribution(arrays_pts, X_Scale, Y_Scale);

	// draw potential theta
	float threshold = 15;// max =255
	cv::Mat color_img = cv::imread("image_roi1.bmp", 1);
	for (int i = 0; i != arrays_pts.size();){
		// find second point from theta
		if (arrays_pts[i] >= threshold){
			//// draw line
			//int temp_x = R1*cos(i*PI / 180);
			//int temp_y = R1*sin(i*PI / 180);
			//cv::Point second_ptr;
			//second_ptr.x = center.x + temp_x;
			//second_ptr.y = center.y + temp_y;
			//cv::Point first_ptr;
			//first_ptr.x = center.x + R2*cos(i*PI / 180);
			//first_ptr.y = center.y + R2*sin(i*PI / 180);
			//cv::line(color_img, first_ptr, second_ptr, cv::Scalar(0,255,0), 1, 0);
			//
			//cv::RotatedRect temp = cv::fitEllipse(array_angle[i]);
			//cv::ellipse(color_img, temp, cv::Scalar(0, 0, 255), 1, 8);
			int numbers = 0;
			int next_idx = i;
			std::vector<cv::Point> total_ptr;
			for (int idx = i; idx != arrays_pts.size(); idx++){
				if (arrays_pts[idx] >= threshold){
					std::copy(array_angle[idx].begin(), array_angle[idx].end(), std::back_inserter(total_ptr));
					numbers += 1;
					next_idx += 1;
				}
				else break;
			}
			// condition
			if (numbers >= 2){
				cv::RotatedRect temp = cv::fitEllipse(total_ptr);
				cv::ellipse(color_img, temp, cv::Scalar(0, 0, 255), 1, 8);
				cv::polylines(color_img, total_ptr, true, cv::Scalar(0, 0, 255), 1, 8);
				i = next_idx;
			}
			else i += 1;
		}
		else i+=1;
	}

	cv::imshow("area for detect", color_img);
	cv::imwrite("area.bmp", color_img);
	return input_img;
}
