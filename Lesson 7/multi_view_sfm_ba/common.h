#ifndef  COMMON_H
#define  COMMON_H
#include <vector>
#include <opencv2/opencv.hpp> 
#include <iostream>

typedef struct CloudPoint
{
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
}CloudPoint;


#endif

