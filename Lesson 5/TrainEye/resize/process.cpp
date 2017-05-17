#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main( int argc, const char** argv )
{
	cv::Mat src=cv::imread(argv[1],1);
	cv::resize(src,src,cv::Size(384,286),0,0,INTER_LINEAR);
	cv::cvtColor( src, src, COLOR_BGR2GRAY );
	cv::imwrite(argv[1],src);
    return 0;
}
