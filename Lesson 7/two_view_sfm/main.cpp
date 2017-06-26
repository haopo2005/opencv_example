#include "triangulation.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#define SHRINK_SCALE 4



int main(int argc, char** argv)
{
	cv::Mat K,Kinv,distortion_coeff,cam_matrix;
	cv::Mat src, dst;
	src = cv::imread(argv[1]);
	dst = cv::imread(argv[2]);
	cv::Mat desc_src, desc_dst, src_small, dst_small;
	std::vector<cv::KeyPoint> kp_src, kp_dst; 
	const double akaze_thresh = 0.0001f; // AKAZE detection threshold set to locate about 1000 keypoints
	cv::Ptr<cv::AKAZE> f2d = cv::AKAZE::create();
	f2d->setThreshold(akaze_thresh);
	
	//compute desc
	cv::resize(src, src_small, cv::Size(src.cols/SHRINK_SCALE, src.rows/SHRINK_SCALE));
	cv::resize(dst, dst_small, cv::Size(dst.cols/SHRINK_SCALE, dst.rows/SHRINK_SCALE));
	f2d->detectAndCompute(src_small, cv::noArray(), kp_src, desc_src);
	f2d->detectAndCompute(dst_small, cv::noArray(), kp_dst, desc_dst);
	
	//hamming match
	cv::BFMatcher matcher(cv::NORM_HAMMING, false);
	std::vector<std::vector< cv::DMatch > > matches;
	matcher.knnMatch(desc_src, desc_dst, matches, 2);
	
	//refine match
	std::vector<cv::DMatch> matches_pure;
	const float minRatio = 0.8f;
	for (size_t i=0; i< matches.size(); i++)     
    {          
	    const cv::DMatch& bestMatch = matches[i][0];         
	    const cv::DMatch& betterMatch = matches[i][1];         
	    float distanceRatio = bestMatch.distance /betterMatch.distance;
	    if (distanceRatio < minRatio)          
	    {             
	        matches_pure.push_back(bestMatch);          
	    } 
	}
	
	//compute f
	std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
    for (size_t i = 0; i < matches_pure.size(); i++)
    {		
      cv::Point2f src = kp_src[matches_pure[i].queryIdx].pt;
      cv::Point2f dst = kp_dst[matches_pure[i].trainIdx].pt;
      srcPoints.push_back(src);
      dstPoints.push_back(dst);
    }
	std::vector<unsigned char> inliersMask(srcPoints.size());
	cv::Mat F = cv::findFundamentalMat(srcPoints, dstPoints, inliersMask, cv::FM_LMEDS);
	std::vector<cv::DMatch> inliers;
	for(size_t i=0;i<inliersMask.size();i++)
	{
	    if(inliersMask[i])
	        inliers.push_back(matches_pure[i]);
	}
	matches_pure.swap(inliers);
	std::cout<<matches_pure.size()<<std::endl;
	std::cout<<F<<std::endl;
	
	cv::Mat img_matches;
	cv::drawMatches(src_small,kp_src,dst_small,kp_dst,matches_pure,img_matches);
	cv::imwrite("test.jpg",img_matches);
	
	read_intrisic(K,Kinv,distortion_coeff,cam_matrix, src_small);
	cv::Mat_<double> E = K.t()*F*K;
	
	cv::Mat_<double> R1(3,3);
	cv::Mat_<double> R2(3,3);
	cv::Mat_<double> t1(1,3);
	cv::Mat_<double> t2(1,3);
	cv::Matx34d P(1,0,0,0,
				  0,1,0,0,
				  0,0,1,0),
				P1(1,0,0,0,
				   0,1,0,0,
				   0,0,1,0);
	
	if(fabsf(determinant(E)) > 1e-05) {
		std::cout << "det(E) != 0 : " << fabsf(determinant(E)) << "\n";
		P1 = 0;
		return false;
	}
	
	//decompse E to P'
	if (!DecomposeEtoRandT(E,R1,R2,t1,t2))
		return false;
	
	if(determinant(R1)+1.0 < 1e-09) 
	{
		//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
		std::cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << std::endl;
		E = -E;
		DecomposeEtoRandT(E,R1,R2,t1,t2);
	}
	if (!CheckCoherentRotation(R1))
	{
		std::cout << "resulting rotation is not coherent\n";
		P1 = 0;
		return false;
	}
	P1 = cv::Matx34d(R1(0,0), R1(0,1), R1(0,2), t1(0),
				 R1(1,0), R1(1,1), R1(1,2), t1(1),
				 R1(2,0), R1(2,1), R1(2,2), t1(2));
	std::cout << "Testing P1 " << std::endl << cv::Mat(P1) << std::endl;
	
	//triangulation
	std::vector<CloudPoint> pcloud, pcloud1;
	std::vector<cv::Point2f> imgpts1_good, imgpts2_good;
	for (size_t i = 0; i < matches_pure.size(); i++)
    {
	  imgpts1_good.push_back(kp_src[matches_pure[i].queryIdx].pt);
	  imgpts2_good.push_back(kp_dst[matches_pure[i].trainIdx].pt);
    }
	double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distortion_coeff, P, P1, pcloud);
	double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distortion_coeff, P1, P, pcloud1);
	std::vector<uchar> tmp_status;
	if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) 
		|| reproj_error1 > 100.0 || reproj_error2 > 100.0) 
	{
		if (!CheckCoherentRotation(R2)) {
			std::cout << "resulting rotation is not coherent\n";
			P1 = 0;
			return false;
		}
					
		P1 = cv::Matx34d(R2(0,0), R2(0,1), R2(0,2), t1(0),
					 R2(1,0), R2(1,1), R2(1,2), t1(1),
					 R2(2,0), R2(2,1), R2(2,2), t1(2));
		std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

		pcloud.clear(); pcloud1.clear();
		reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distortion_coeff, P, P1, pcloud);
		reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distortion_coeff, P1, P, pcloud1);
					
		if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) 
			|| reproj_error1 > 100.0 || reproj_error2 > 100.0) 
		{
			P1 = cv::Matx34d(R2(0,0), R2(0,1), R2(0,2), t1(0),
					 R2(1,0), R2(1,1), R2(1,2), t1(1),
					 R2(2,0), R2(2,1), R2(2,2), t1(2));
			std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

			pcloud.clear(); pcloud1.clear(); 
			reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distortion_coeff, P, P1, pcloud);
			reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distortion_coeff, P1, P, pcloud1);
			
			if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) 
				|| reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				std::cout << "Shit." << std::endl; 
				//return false;
			}
		}				
	}			
	
	//imgpts1_good && pcloud		
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud.width = imgpts2_good.size();
	cloud.height = 1;
	cloud.is_dense = true;
	cloud.points.resize(cloud.width * cloud.height);
	
	for(unsigned int k=0; k < imgpts1_good.size();k++)
	{
		pcl::PointXYZRGB pclp;
		int x = imgpts2_good[k].x;
		int y = imgpts2_good[k].y;
		
		pclp.x = pcloud1[k].pt.x;
		pclp.y = pcloud1[k].pt.y;
		pclp.z = pcloud1[k].pt.z;
		pclp.b = dst_small.at<cv::Vec3b>(y,x)[0];
		pclp.g = dst_small.at<cv::Vec3b>(y,x)[1];
		pclp.r = dst_small.at<cv::Vec3b>(y,x)[2];
		//std::cout<<"x:"<<pclp.x<<",y:"<<pclp.y<<",z:"<<pclp.z<<std::endl;
		
		cloud.push_back(pclp);
	}
			
	
	pcl::io::savePLYFileBinary("color.ply", cloud);
	return 0;
}
