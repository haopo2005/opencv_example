#ifndef  MATCHING_H
#define  MATCHING_H
#include <iostream>
#include <vector>
#include <map>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/xfeatures2d.hpp>

#define SHRINK_SCALE 2

class MatchingImageList
{
private:	
	std::vector<cv::Mat> descriptors_;
	void draw(int left, int right, const std::vector<cv::KeyPoint> left_kp, const std::vector<cv::KeyPoint> right_kp,const std::vector<cv::DMatch> matches);

public:
	bool ReadFromDirectory(const std::string dir_path);
	std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);
	bool MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>& matches);
	bool MatchImageListsFeatures(const std::vector<std::string> imglist);	

	cv::Size imgs_size;
	std::vector<std::vector<cv::KeyPoint> > imgpts;
	std::vector<std::string> imglist;
	std::map<std::pair<int,int>, std::vector<cv::DMatch> > matches_matrix;
	std::vector<cv::Mat> imgs_orig;
};
#endif
