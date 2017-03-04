#include <opencv2/opencv.hpp> 
#include <iostream>
#include <vector>

void findContoursAndSort(const cv::Mat& thresholdImg, std::vector<std::vector<cv::Point> >& contours)  
{   
    std::vector<cv::Point> temp;
	//从二值图中提取轮廓
	cv::findContours(thresholdImg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);      
	
	//假设图片中就一张发票，因此提取轮廓后，对轮廓面积进行排序
	for(size_t i=0;i<contours.size() - 1;i++)
	{
	    for(size_t j=0;j<contours.size() - 1 - i;j++)
		{
		    if (cv::contourArea(cv::Mat(contours[j])) < cv::contourArea(cv::Mat(contours[j + 1])))
			{
				temp = contours[j + 1];
				contours[j + 1] = contours[j];
				contours[j] = temp;
			}
		}
	}
} 

std::vector<cv::Point> findRect(std::vector<std::vector<cv::Point> > contours, int &found)
{
    std::vector<std::vector<cv::Point> > closed_contours;
	closed_contours.resize(contours.size());
	for(size_t i = 0;i < contours.size();i++)
	{
	    //获取轮廓长度
	    double peri =  cv::arcLength(cv::Mat(contours[i]), true);
		//逼近多边形	
		cv::approxPolyDP(cv::Mat(contours[i]), closed_contours[i], 0.08*peri, true);

		//如果最先逼近的多边形只有四个顶点，则认为是矩形
		if(closed_contours[i].size() == 4)
		{
		    found = 1;
		    return closed_contours[i];
		}
	}
} 

std::vector<cv::Point> OrderPoints(std::vector<cv::Point> m_rect)
{
	//0----1
	//|    |
	//3----2
	//发票0左上角x与y之和最小，右下角2x与y之和最大
	//发票1右上角x与y之差最小，左下角3x与y之差最大
	std::vector<int> sum;
	std::vector<int> diff;
	for(size_t i = 0;i < m_rect.size();i++)
	{
	    sum.push_back(m_rect[i].x + m_rect[i].y);
		diff.push_back(m_rect[i].x - m_rect[i].y);
	}
	auto sum_max = std::max_element(std::begin(sum), std::end(sum));
    auto sum_min = std::min_element(std::begin(sum), std::end(sum));  	
    auto diff_max = std::max_element(std::begin(diff), std::end(diff));
    auto diff_min = std::min_element(std::begin(diff), std::end(diff));
	
	std::vector<cv::Point> final_rect;
	final_rect.resize(m_rect.size());
	int tl = std::distance(std::begin(sum), sum_min);
	int bl = std::distance(std::begin(diff), diff_min);
	int br = std::distance(std::begin(sum), sum_max);
	int tr = std::distance(std::begin(diff), diff_max);
	final_rect[0] = m_rect[tl];
	final_rect[1] = m_rect[tr];
	final_rect[2] = m_rect[br];
	final_rect[3] = m_rect[bl];
	
	return final_rect;
}

cv::Mat FourPointTransform(cv::Mat image, std::vector<cv::Point> m_rect)
{
    //计算矩形顶点
	std::vector<cv::Point> real_rect = OrderPoints(m_rect);
	
	//计算矩形边长
	float widthA = sqrt((real_rect[0].x - real_rect[1].x)*(real_rect[0].x - real_rect[1].x)
	           +(real_rect[0].y - real_rect[1].y)*(real_rect[0].y - real_rect[1].y));
	float widthB = sqrt((real_rect[3].x - real_rect[2].x)*(real_rect[3].x - real_rect[2].x)
	           +(real_rect[3].y - real_rect[2].y)*(real_rect[3].y - real_rect[2].y));
			   
	int maxWidth = std::max(int(widthA), int(widthB));
	
	float heightA = sqrt((real_rect[0].x - real_rect[3].x)*(real_rect[0].x - real_rect[3].x)
	           +(real_rect[0].y - real_rect[3].y)*(real_rect[0].y - real_rect[3].y));
	float heightB = sqrt((real_rect[1].x - real_rect[2].x)*(real_rect[1].x - real_rect[2].x)
	           +(real_rect[1].y - real_rect[2].y)*(real_rect[1].y - real_rect[2].y));
			   
	int maxHeight = std::max(int(heightA), int(heightB));
	
	cv::Point2f s[4], d[4];
	s[0] = cv::Point2f(real_rect[0].x, real_rect[0].y);
    s[1] = cv::Point2f(real_rect[1].x, real_rect[1].y);
    s[2] = cv::Point2f(real_rect[2].x, real_rect[2].y);
    s[3] = cv::Point2f(real_rect[3].x, real_rect[3].y);
    
	d[0] = cv::Point2f(0, 0);
    d[1] = cv::Point2f(maxWidth, 0);
	d[2] = cv::Point2f(maxWidth, maxHeight);
	d[3] = cv::Point2f(0, maxHeight);
	
	cv::Mat transform = cv::getPerspectiveTransform(s, d);
	std::cout<<"perspective transform matrix:"<<std::endl;
	std::cout<<transform<<std::endl;
	
	cv::Mat dst_image;
	cv::warpPerspective(image, dst_image, transform, cv::Size(maxWidth,maxHeight));
	return dst_image;
}

int main(int argc, char **argv)
{
    //载入发票图片的灰度图
    cv::Mat src, grey, small_src, edges;
    src	= cv::imread(argv[1]);
	
	//等比例缩小，缩小比例不当或者拍摄问题，都会导致矩形提取失败！
	
	/*double scale = 0.5;
	float height = src.rows*scale;
	float width = src.cols*scale;
	cv::resize(src, small_src, cv::Size(height, width));*/
	float width = 1000.0;
	float scale = width/src.cols;
	float height = src.rows*scale;
	cv::resize(src, small_src, cv::Size(height, width));
	
	//small_src = src;
	cv::cvtColor(small_src, grey, CV_BGR2GRAY);
	
	//高斯模糊,平滑窗口大小5*5,sigma_x和sigma_y均为0
	cv::GaussianBlur(grey, grey, cv::Size(5,5), 0, 0);
	
	//提取canny边缘
	cv::Canny(grey, edges, 75, 200);
	
	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point> > m_rect;
	findContoursAndSort(edges, contours);
	int found = 0;
	m_rect.push_back(findRect(contours, found));
	if(!found)
	{
	    std::cout<<" not able to find the rectangle, take another jpg "<<std::endl;
		return 0;
	}
	
	//投影变换获得鸟瞰图
	cv::Mat dst = FourPointTransform(small_src, m_rect[0]);
	cv::imwrite("warp.jpg",dst);
	//二值化处理
	cv::Mat dst_grey, dst_binary;
	cv::cvtColor(dst, dst_grey, CV_BGR2GRAY);
	cv::adaptiveThreshold(dst_grey, dst_binary, 255,
                        CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
    cv::imwrite("final.jpg", dst_binary);
    return 0;
}