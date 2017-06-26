#include "MatchingImageList.h"


//读取文件夹内所有图片，放入imglist数组中
bool MatchingImageList::ReadFromDirectory(const std::string dir_path)
{
    boost::filesystem::path images_dir(dir_path);
    boost::filesystem::recursive_directory_iterator iter(images_dir), eod;
    std::vector<int> area;
    int count = 0;
	
    BOOST_FOREACH(boost::filesystem::path const& i, std::make_pair(iter, eod)) {
      if (boost::filesystem::is_regular_file(i)
          && (i.extension().string() == ".jpg"
          || i.extension().string() == ".JPG" || i.extension().string() == ".png")){
        std::cout<< "Adding image " << i.string() <<std::endl;
		imglist.push_back(i.string());
      }
    }

	//调试项
	for(auto pic_name:imglist)
	{
		std::cout<<pic_name<<std::endl;
	}
	return true;
}

//互换matches中queryIdx和trainIdx索引
std::vector<cv::DMatch> MatchingImageList::FlipMatches(const std::vector<cv::DMatch>& matches) {
	std::vector<cv::DMatch> flip;
	for(int i=0;i<matches.size();i++) {
		flip.push_back(matches[i]);
		std::swap(flip.back().queryIdx,flip.back().trainIdx);
	}
	return flip;
}

//提取两幅图之间的匹配点
bool MatchingImageList::MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>& matches) 
{
	const std::vector<cv::KeyPoint>& imgpts1 = imgpts[idx_i];
    const std::vector<cv::KeyPoint>& imgpts2 = imgpts[idx_j];
    const cv::Mat& descriptors_1 = descriptors_[idx_i];
    const cv::Mat& descriptors_2 = descriptors_[idx_j];

	std::vector<cv::DMatch> good_matches_,very_good_matches_;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

	keypoints_1 = imgpts1;
    keypoints_2 = imgpts2;

	if(descriptors_1.empty()) {
        std::cout<<"descriptors_1 is empty"<<std::endl;
    }
    if(descriptors_2.empty()) {
        std::cout<<"descriptors_2 is empty"<<std::endl;
    }

	cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector< cv::DMatch > > matches2;	
    matcher.knnMatch(descriptors_1, descriptors_2, matches2, 2);
	const float minRatio = 0.8f;	
	std::vector<cv::DMatch> d_match;
	for (size_t i=0; i< matches2.size(); i++)     
    {          
	    const cv::DMatch& bestMatch = matches2[i][0];         
	    const cv::DMatch& betterMatch = matches2[i][1];         
	    float distanceRatio = bestMatch.distance /betterMatch.distance;
	    if (distanceRatio < minRatio)          
	    {             
	        d_match.push_back(bestMatch);          
	    } 
	}
	matches.swap(d_match);

	//利用极性约束过滤匹配点
	std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
    for (size_t i = 0; i < matches.size(); i++)
    {		
      cv::Point2f src = imgpts[idx_i][matches[i].queryIdx].pt;
      cv::Point2f dst = imgpts[idx_j][matches[i].trainIdx].pt;
      srcPoints.push_back(src);
      dstPoints.push_back(dst);
    }
	std::vector<unsigned char> inliersMask(srcPoints.size());
	cv::Mat homography = cv::findFundamentalMat(srcPoints, dstPoints,inliersMask, cv::FM_LMEDS);
	std::vector<cv::DMatch> inliers;	
	for (size_t i=0; i<inliersMask.size(); i++)	
	{		
	    if (inliersMask[i])			
	      inliers.push_back(matches[i]);	
	}	
	matches.swap(inliers);
	return true;
}


//遍历图像列表，两两匹配
bool MatchingImageList::MatchImageListsFeatures(const std::vector<std::string> imglist)
{
   const double akaze_thresh = 0.0005f; // AKAZE 阈值，可以检测1000个以上特征点
   cv::Ptr<cv::AKAZE> f2d = cv::AKAZE::create();
   f2d->setThreshold(akaze_thresh);

   //遍历图像，提取每幅图像特征点
   for (const std::string& img_file : imglist) 
   {
       cv::Mat img, img_small, img_blur, desc;
       img = cv::imread(img_file);
	   std::cout<< "Loading image" <<img_file<<std::endl;
	   
	   //将图像缩放到合适尺寸，加速运算
	   cv::resize(img, img_small, cv::Size(img.cols/SHRINK_SCALE, img.rows/SHRINK_SCALE));
	   imgs_size = img_small.size();

	   //保存图片颜色到imgs_orig数组中
	   imgs_orig.push_back(img_small);
	   
	   //提取特征点及特征描述
	   std::vector<cv::KeyPoint> kp;    
	   f2d->detectAndCompute(img_small, cv::noArray(), kp, desc);
	   descriptors_.push_back(desc);
	   imgpts.push_back(kp);
   } 

    //遍历图片列表，两两匹配
    int loop1_top = imglist.size() - 1, loop2_top = imglist.size();
    int frame_num_i = 0;
	for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) 
	{
		for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
		{
			std::cout << "------------ Matching " << imglist[frame_num_i] << ","<<imglist[frame_num_j]<<" ------------\n";
			std::vector<cv::DMatch> matches_tmp;
			MatchFeatures(frame_num_i,frame_num_j,matches_tmp);
			matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;
			
			std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
			matches_matrix[std::make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip;

			//draw(frame_num_i, frame_num_j, imgpts[frame_num_i], imgpts[frame_num_j], matches_tmp);
		}
	}
	return true;
}

//查看匹配结果，调试用
void MatchingImageList::draw(int left, int right, const std::vector<cv::KeyPoint> left_kp, const std::vector<cv::KeyPoint> right_kp, const std::vector<cv::DMatch> matches)
{
    char dst[30];
    cv::Mat img_matches,img1_small,img2_small;    
	cv::Mat img1 = cv::imread(imglist[left]);
	cv::Mat img2 = cv::imread(imglist[right]);
	cv::resize(img1, img1_small, cv::Size(img1.cols/SHRINK_SCALE, img1.rows/SHRINK_SCALE));
    cv::resize(img2, img2_small, cv::Size(img1.cols/SHRINK_SCALE, img1.rows/SHRINK_SCALE));
	cv::drawMatches(img1_small, left_kp, img2_small, right_kp, matches, img_matches); 
	sprintf(dst, "/root/result/%03d_%03d.jpg", left, right);
	cv::imwrite(dst, img_matches);
}


