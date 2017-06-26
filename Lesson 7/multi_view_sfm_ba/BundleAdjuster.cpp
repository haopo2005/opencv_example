#include "BundleAdjuster.h"
#define V3DLIB_ENABLE_SUITESPARSE true

#include <Math/v3d_linear.h>
#include <Base/v3d_vrmlio.h>
#include <Geometry/v3d_metricbundle.h>

using namespace V3D;
using namespace std;
using namespace cv;

namespace{
	inline void showErrorStatistics(double const f0,
							V3D::StdDistortionFunction const& distortion,
							std::vector<V3D::CameraMatrix> const& cams,
							std::vector<V3D::Vector3d> const& Xs,
							std::vector<V3D::Vector2d> const& measurements,
							std::vector<int> const& correspondingView,
							std::vector<int> const& correspondingPoint)
	{
		int const K = measurements.size();		
		double meanReprojectionError = 0.0;

		for (int k = 0; k < K; ++k)
		{
			int const i = correspondingView[k];
			int const j = correspondingPoint[k];
			V3D::Vector2d p = cams[i].projectPoint(distortion, Xs[j]);

			double reprojectionError = V3D::norm_L2(f0 * (p - measurements[k]));
			meanReprojectionError += reprojectionError;
		}
		std::cout << "mean reprojection error (in pixels): " << meanReprojectionError/K << std::endl;
	}
}

//count number of 2D measurements
int BundleAdjuster::Count2DMeasurements(const std::vector<CloudPoint>& pointcloud) 
{
	int K = 0;
	for (unsigned int i=0; i<pointcloud.size(); i++) {
		for (unsigned int ii=0; ii<pointcloud[i].imgpt_for_img.size(); ii++) {
			if (pointcloud[i].imgpt_for_img[ii] >= 0) {
				K ++;
			}
		}
	}
	return K;
}


void BundleAdjuster::adjustBundle(std::vector<CloudPoint>& pointcloud, 
								  cv::Mat& cam_matrix,
								  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
								  std::map<int ,cv::Matx34d>& Pmats
								) 
{
	int N = Pmats.size(), M = pointcloud.size(), K = Count2DMeasurements(pointcloud);
	
	std::cout << "N (cams) = " << N << " M (points) = " << M << " K (measurements) = " << K << std::endl;
	
	V3D::StdDistortionFunction distortion;
	
	//conver camera intrinsics to BA datastructs
	V3D::Matrix3x3d KMat;
	V3D::makeIdentityMatrix(KMat);
	KMat[0][0] = cam_matrix.at<double>(0,0); //fx
	KMat[1][1] = cam_matrix.at<double>(1,1); //fy
	KMat[0][1] = cam_matrix.at<double>(0,1); //skew
	KMat[0][2] = cam_matrix.at<double>(0,2); //ppx
	KMat[1][2] = cam_matrix.at<double>(1,2); //ppy
	
	double const f0 = KMat[0][0];
	std::cout << "intrinsic before bundle = ";
	V3D::displayMatrix(KMat);
	V3D::Matrix3x3d Knorm = KMat;

	// Normalize the intrinsic to have unit focal length.
	V3D::scaleMatrixIP(1.0/f0, Knorm);
	Knorm[2][2] = 1.0;
	
	std::vector<int> pointIdFwdMap(M);
	std::map<int, int> pointIdBwdMap;
	
	//conver 3D point cloud to BA datastructs
	std::vector<V3D::Vector3d > Xs(M);
	for (int j = 0; j < M; ++j)
	{
		int pointId = j;
		Xs[j][0] = pointcloud[j].pt.x;
		Xs[j][1] = pointcloud[j].pt.y;
		Xs[j][2] = pointcloud[j].pt.z;
		pointIdFwdMap[j] = pointId;
		pointIdBwdMap.insert(std::make_pair(pointId, j));
	}
	std::cout << "Read the 3D points." << std::endl;
	
	std::vector<int> camIdFwdMap(N,-1);
	std::map<int, int> camIdBwdMap;
	
	//convert cameras to BA datastructs
	std::vector<V3D::CameraMatrix> cams(N);
	for (int i = 0; i < N; ++i)
	{
		int camId = i;
		V3D::Matrix3x3d R;
		V3D::Vector3d T;
		
		cv::Matx34d& P = Pmats[i];
		
		R[0][0] = P(0,0); R[0][1] = P(0,1); R[0][2] = P(0,2); T[0] = P(0,3);
		R[1][0] = P(1,0); R[1][1] = P(1,1); R[1][2] = P(1,2); T[1] = P(1,3);
		R[2][0] = P(2,0); R[2][1] = P(2,1); R[2][2] = P(2,2); T[2] = P(2,3);
		
		camIdFwdMap[i] = camId;
		camIdBwdMap.insert(std::make_pair(camId, i));
		
		cams[i].setIntrinsic(Knorm);
		cams[i].setRotation(R);
		cams[i].setTranslation(T);
	}
	std::cout << "Read the cameras." << std::endl;
	
	std::vector<V3D::Vector2d > measurements;
	std::vector<int> correspondingView;
	std::vector<int> correspondingPoint;
	
	measurements.reserve(K);
	correspondingView.reserve(K);
	correspondingPoint.reserve(K);
	
	//convert 2D measurements to BA datastructs
	for (unsigned int k = 0; k < pointcloud.size(); ++k)
	{
		for (unsigned int i=0; i<pointcloud[k].imgpt_for_img.size(); i++) {
			if (pointcloud[k].imgpt_for_img[i] >= 0) {
				int view = i, point = k;
				V3D::Vector3d p, np;
				
				cv::Point cvp = imgpts[i][pointcloud[k].imgpt_for_img[i]].pt;
				p[0] = cvp.x;
				p[1] = cvp.y;
				p[2] = 1.0;
				
				if (camIdBwdMap.find(view) != camIdBwdMap.end() &&
					pointIdBwdMap.find(point) != pointIdBwdMap.end())
				{
					// Normalize the measurements to match the unit focal length.
					V3D::scaleVectorIP(1.0/f0, p);
					measurements.push_back(V3D::Vector2d(p[0], p[1]));
					correspondingView.push_back(camIdBwdMap[view]);
					correspondingPoint.push_back(pointIdBwdMap[point]);
				}
			}
		}
	} // end for (k)
	
	K = measurements.size();
	
	std::cout << "Read " << K << " valid 2D measurements." << std::endl;
	
	showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

//	V3D::optimizerVerbosenessLevel = 1;
	double const inlierThreshold = 2.0 / fabs(f0);
	
	V3D::Matrix3x3d K0 = cams[0].getIntrinsic();
	std::cout << "K0 = "; 
	V3D::displayMatrix(K0);

	bool good_adjustment = false;
	
	V3D::ScopedBundleExtrinsicNormalizer extNorm(cams, Xs);
	V3D::ScopedBundleIntrinsicNormalizer intNorm(cams,measurements,correspondingView);
	V3D::CommonInternalsMetricBundleOptimizer opt(V3D::FULL_BUNDLE_FOCAL_LENGTH_PP, inlierThreshold, K0, distortion, cams, Xs,
											 measurements, correspondingView, correspondingPoint);
//		StdMetricBundleOptimizer opt(inlierThreshold,cams,Xs,measurements,correspondingView,correspondingPoint);
	
	opt.tau = 1e-3;
	opt.maxIterations = 50;
	opt.minimize();
	
	std::cout << "optimizer status = " << opt.status << std::endl;
	
	good_adjustment = (opt.status != 2);
	
	
	std::cout << "refined K = ";
	V3D::displayMatrix(K0);
	
	for (int i = 0; i < N; ++i) cams[i].setIntrinsic(K0);
	
	V3D::Matrix3x3d Knew = K0;
	V3D::scaleMatrixIP(f0, Knew);
	Knew[2][2] = 1.0;
	std::cout << "Knew = ";
	V3D::displayMatrix(Knew);
	
	showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);
	
	if(good_adjustment) { //good adjustment?		
		//extract 3D points
		for (unsigned int j = 0; j < Xs.size(); ++j)
		{
			//if (distance_L2(Xs[j], mean) > 3*distThr) makeZeroVector(Xs[j]);
			
			pointcloud[j].pt.x = Xs[j][0];
			pointcloud[j].pt.y = Xs[j][1];
			pointcloud[j].pt.z = Xs[j][2];
		}
		
		//extract adjusted cameras
		for (int i = 0; i < N; ++i)
		{
			V3D::Matrix3x3d R = cams[i].getRotation();
			V3D::Vector3d T = cams[i].getTranslation();
			
			cv::Matx34d P;
			P(0,0) = R[0][0]; P(0,1) = R[0][1]; P(0,2) = R[0][2]; P(0,3) = T[0];
			P(1,0) = R[1][0]; P(1,1) = R[1][1]; P(1,2) = R[1][2]; P(1,3) = T[1];
			P(2,0) = R[2][0]; P(2,1) = R[2][1]; P(2,2) = R[2][2]; P(2,3) = T[2];
			
			Pmats[i] = P;
		}
		

		//TODO: extract camera intrinsics
		cam_matrix.at<double>(0,0) = Knew[0][0];
		cam_matrix.at<double>(0,1) = Knew[0][1];
		cam_matrix.at<double>(0,2) = Knew[0][2];
		cam_matrix.at<double>(1,1) = Knew[1][1];
		cam_matrix.at<double>(1,2) = Knew[1][2];
	}
}


