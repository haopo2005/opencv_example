#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void ComputeHist(const cv::Mat labels, std::vector<float> &counter, int height, int width, int nClusters)
{
	
	unsigned int sumHist = 0;
	
	for(unsigned int i = 0;i < height * width; i++)
	{
		int  clusterIndex = labels.at<int>(i,0);
		for(int j = 0;j < nClusters;j++)
		{
			if(clusterIndex == j)
			{
				counter[clusterIndex]++;
				sumHist++;
			}
		}
	}
	
	for(int k = 0;k < nClusters; k++)
	{
		counter[k] /= sumHist;
		std::cout<<counter[k]<<std::endl;
	}
}


void drawHist(std::vector<float> mhist, const cv::Mat centers, int nClusters)
{
	float startX, endX;
	startX = 0;
	endX = 0;
	cv::Mat resultImg(50,300,CV_8UC3);
	
	for(int i = 0;i < nClusters;i++)
	{
		endX = startX + mhist[i]*300;
		cv::Scalar centerColor;
		centerColor.val[0] = (int)centers.at<float>(i,0);
		centerColor.val[1] = (int)centers.at<float>(i,1);
		centerColor.val[2] = (int)centers.at<float>(i,2);
		cv::rectangle(resultImg, cv::Point(startX, 0), cv::Point(endX, 50), centerColor, CV_FILLED);
		std::cout<<"startX:"<<startX<<",endX"<<endX<<std::endl;
		startX = endX;
	}

	cv::imwrite("resultImg.jpg", resultImg);
}

int main(int argc, char** argv)
{
	cv::Mat img = cv::imread(argv[1]);
	//cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	
	std::vector<cv::Mat> rgb;
	cv::split(img, rgb);
	cv::Mat mdata(img.rows*img.cols, 3, CV_32F);	
	
	for(unsigned int i=0; i < img.cols*img.rows; i++) 
    {
        mdata.at<float>(i,0) = (float)rgb[0].data[i];        
        mdata.at<float>(i,1) = (float)rgb[1].data[i];        
        mdata.at<float>(i,2) = (float)rgb[2].data[i];
    }
	
	int nClusters = atoi(argv[2]);
	cv::Mat labels, centers;
	std::vector<float> counter(nClusters, 0);
	cv::kmeans(mdata, nClusters, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0),
               5,cv::KMEANS_PP_CENTERS, centers);

	ComputeHist(labels, counter,img.rows, img.cols, nClusters);
	
	drawHist(counter, centers, nClusters);
	return 0;
}