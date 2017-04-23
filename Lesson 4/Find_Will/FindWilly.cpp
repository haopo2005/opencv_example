#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
	cv::Mat src = cv::imread(argv[1]);
	cv::Mat dst = cv::imread(argv[2]);
	cv::Mat result;
	cv::Point minLoc, maxLoc;
	
	cv::matchTemplate(src, dst, result, cv::TM_CCOEFF);
	cv::minMaxLoc(result, NULL, NULL, &minLoc, &maxLoc);
	
	cv::Mat mask(src.rows, src.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat final_result;
	cv::addWeighted(src, 0.25, mask, 0.75, 0, final_result, -1);
	cv::Mat roi2 = final_result(cv::Rect(maxLoc.x, maxLoc.y, dst.cols, dst.rows));
	cv::Mat roi1 = src(cv::Rect(maxLoc.x, maxLoc.y, dst.cols, dst.rows));
	roi1.copyTo(roi2);
	cv::imwrite("test.jpg", final_result);
	return 0;
}