#include "MaximaLocalMethod.h"



MaximaLocalMethod::MaximaLocalMethod()
{
}


MaximaLocalMethod::~MaximaLocalMethod()
{
}

void MaximaLocalMethod::InvertGray255(cv::Mat & input_img, cv::Mat & output_img)
{
	assert(input_img.channels() != 3);
	float R1 = (float)input_img.cols / 2 ;
	cv::Point center = cv::Point(input_img.cols / 2, input_img.rows / 2);
	float R2 = (float)input_img.cols / 2 - 90 ;
	output_img = input_img.clone();
	for (unsigned int i = 0; i != input_img.rows; i++) {
		uchar *Mi = input_img.ptr<uchar>(i);
		uchar *Mo = output_img.ptr<uchar>(i);
		for (unsigned int j = 0; j != input_img.cols; j++) {

			cv::Point point_pt = cv::Point(j, i);
			int temp_dis = cv::norm(point_pt - center);
			if (temp_dis >= R1 || temp_dis <=R2) Mo[j] = 0;
			else Mo[j] = 255 - Mi[j];
		}
	}
}

void MaximaLocalMethod::InvertGray255SSE(cv::Mat & input_img, cv::Mat & output_img)
{
	assert(input_img.channels() != 3);
	output_img = input_img.clone();
	uchar * Mi = (uchar*)cv::alignPtr(input_img.data,16);
	uchar * Mo = (uchar*)cv::alignPtr(output_img.data,16);

	const __m128i temple = _mm_set1_epi8(255);
	const __m128i gamma = _mm_set1_epi8(50);
	__m128i input_ptr, output_ptr;
	for (unsigned int i = 0; i < input_img.cols*input_img.rows; i += 16) {
		input_ptr =  _mm_load_si128((__m128i*)&Mi[i]);
		output_ptr = _mm_subs_epi8(temple, input_ptr);
		output_ptr = _mm_adds_epi8(gamma, output_ptr);
		_mm_store_si128((__m128i*)&Mo[i], output_ptr);
	}

	//_mm_free(Mi);
	//_mm_free(Mo);
}

void MaximaLocalMethod::InvertGray255AVX(cv::Mat & input_img, cv::Mat & output_img)
{
	assert(input_img.channels() != 3);
	output_img = input_img.clone();
	uchar * Mi = (uchar*)cv::alignPtr(input_img.data, 32);
	uchar * Mo = (uchar*)cv::alignPtr(output_img.data, 32);

	const __m256i temple = _mm256_set1_epi8(255);
	__m256i input_ptr, output_ptr;
	for (unsigned int i = 0; i < input_img.cols*input_img.rows; i += 32) {
		input_ptr = _mm256_load_si256((__m256i*)&Mi[i]);
		output_ptr = _mm256_subs_epi8(temple, input_ptr);
		_mm256_store_si256((__m256i*)&Mo[i], output_ptr);
	}
}
