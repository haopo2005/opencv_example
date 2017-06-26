#ifndef  TRIANGLE
#define  TRIANGLE
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> 
#include <Eigen/Eigen>
#include "common.h"
#define EPSILON 0.0001





class Triangle
{	
private:
	std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);
public:
	void TakeSVDOfE(cv::Mat_<double>& E, cv::Mat& svd_u, cv::Mat& svd_vt, cv::Mat& svd_w);
	
	bool DecomposeEtoRandT(cv::Mat_<double>& E,cv::Mat_<double>& R1,cv::Mat_<double>& R2,cv::Mat_<double>& t1,cv::Mat_<double>& t2);

	bool CheckCoherentRotation(cv::Mat_<double>& R);

	cv::Mat_<double> LinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);

	cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,cv::Matx34d P,cv::Point3d u1,cv::Matx34d P1);

	bool TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, std::vector<uchar>& status);
};

#endif
