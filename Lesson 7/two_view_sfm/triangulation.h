#include <opencv2/opencv.hpp> 
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#define EPSILON 0.0001

typedef struct CloudPoint{
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
}CloudPoint;

void read_intrisic(cv::Mat& K, cv::Mat& Kinv, cv::Mat& distortion_coeff, cv::Mat& cam_matrix, cv::Mat src);

void TakeSVDOfE(cv::Mat_<double>& E, cv::Mat& svd_u, cv::Mat& svd_vt, cv::Mat& svd_w);

bool DecomposeEtoRandT(cv::Mat_<double>& E,cv::Mat_<double>& R1,cv::Mat_<double>& R2,cv::Mat_<double>& t1,cv::Mat_<double>& t2);

bool CheckCoherentRotation(cv::Mat_<double>& R);

cv::Mat_<double> LinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1);

cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,cv::Matx34d P,cv::Point3d u1,cv::Matx34d P1);

double TriangulatePoints(const std::vector<cv::Point2f>& _pt_set1_pt,const std::vector<cv::Point2f>& _pt_set2_pt,
						const cv::Mat& K,const cv::Mat& Kinv,const cv::Mat& distcoeff,const cv::Matx34d& P,
						const cv::Matx34d& P1,std::vector<CloudPoint>& pointcloud);

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);

bool TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, std::vector<uchar>& status);
