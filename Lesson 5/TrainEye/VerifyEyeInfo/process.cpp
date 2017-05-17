#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main( int argc, char *argv[] )
{
	FILE *fp;
	char img_name[100];
	char dst_name[100];
	int lx,ly,rx,ry,h,w,num;
	
	fp = fopen(argv[1],"r");
	long jj=0;
	while(1)
	{
		fscanf(fp,"%s %d %d %d %d %d %d %d %d %d",img_name, &num, &lx, &ly, &w, &h, &rx, &ry, &w, &h);
		if (feof(fp)) break;
		cv::Mat src=cv::imread(img_name,1);
		cv::rectangle(src,cv::Rect(lx,ly,w,h),cv::Scalar(0,0,255),1,1,0);
		cv::rectangle(src,cv::Rect(rx,ry,w,h),cv::Scalar(0,0,255),1,1,0);
		sprintf(dst_name,"/home/jst/share/TrainEye/VerifyEyeInfo_Result/%d.jpg",jj++);
		cv::imwrite(dst_name,src);
	} 
    return 0;
}
