#ifndef  BA_H
#define  BA_H
#include "common.h"
#include <vector>
#include <opencv2/opencv.hpp> 
#include <iostream>

class BundleAdjuster {
public:
	void adjustBundle(std::vector<CloudPoint>& pointcloud, 
					  cv::Mat& cam_matrix,
					  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
					  std::map<int ,cv::Matx34d>& Pmats);
private:
	int Count2DMeasurements(const std::vector<CloudPoint>& pointcloud);
};

#endif
