#ifndef  MULTICAMERA
#define  MULTICAMERA
#include "MatchingImageList.h"
#include "Triangle.h"
#include "BundleAdjuster.h"
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <string>

#include <opencv2/opencv.hpp> 
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

bool sort_by_first(std::pair<int,std::pair<int,int> > a, std::pair<int,std::pair<int,int> > b);

class MultiCamera
{
public:
	void read_intrisic();
	
	void GetBaseLineTriangulation();
	
	bool FindCameraMatrices(const int first_view, const int second_view, 
						cv::Matx34d& P, cv::Matx34d& P1, std::vector<cv::DMatch>& matches);

	void LetsGo(const std::string dir_path);

	double TriangulatePoints(const std::vector<cv::Point2f>& _pt_set1_pt,const std::vector<cv::Point2f>& _pt_set2_pt,
						const cv::Matx34d& P,const cv::Matx34d& P1,std::vector<CloudPoint>& pointcloud);

	bool FindPoseEstimation(int working_view, cv::Mat_<double>& rvec, cv::Mat_<double>& t, cv::Mat_<double>& R, std::vector<cv::Point3f> ppcloud, std::vector<cv::Point2f> imgPoints);

	std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);	

	bool TriangulatePointsBetweenViews(const int working_view, const int older_view, std::vector<CloudPoint>& new_triangulated, std::vector<int>& add_to_cloud); 
	
	void GetRGBForPointCloud(std::vector<cv::Vec3b>& RGBforCloud);

	void Find2D3DCorrespondences(int working_view, std::vector<cv::Point3f>& ppcloud, std::vector<cv::Point2f>& imgPoints);
	
private:
	Triangle ComputeTriangle;
	MatchingImageList MyMatch;
	BundleAdjuster myBA;

	int m_first_view;
	int m_second_view;
	std::map<int,cv::Matx34d> Pmats;
	std::set<int> done_views;
	std::set<int> good_views;
	std::vector<CloudPoint> all_pcloud;
	cv::Mat K, Kinv, distortion_coeff, cam_matrix;
};

#endif
