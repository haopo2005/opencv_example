#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp> 
#include <iostream>
#include <vector>


int main(int argc, char **argv)
{
    cv::Mat frame; 
	long frameCounter = 0;
	bool stop(false);
	//定义HSV空间肤色上下界
	cv::Scalar lower(10, 50, 100);
	cv::Scalar upper(20, 255, 255);
	//定义计时器
    time_t start, end;
	
	cv::VideoCapture capture(argv[1]);
	if (!capture.isOpened())
		return 1;

	//获取原始视频帧率
	double rate= capture.get(CV_CAP_PROP_FPS);
	int delay= 1000/rate;
	
	cv::namedWindow("Skin Detect Example", CV_WINDOW_NORMAL);
	//定义椭圆结构元素
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
	
	//计时器开始
    time(&start);
	
	while (!stop) {
		char string[20];	
	    cv::Mat hsv, skinmask, skin, hcon;
		//读取当前图像
		if (!capture.read(frame))
			break;
		
		//BGR转换到HSV
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
		//根据肤色上下界，计算肤色模版，0表示非肤色，255表示肤色
		cv::inRange(hsv, lower, upper, skinmask);
		//按照椭圆形状进行腐蚀膨胀操作去除监测肤色的outiler
		cv::dilate(skinmask,skinmask,kernel,cv::Point(-1,-1),2);
		cv::erode(skinmask,skinmask,kernel,cv::Point(-1,-1),3);
		//去除肤色噪点
		cv::GaussianBlur(skinmask,skinmask,cv::Size(3,3),0);
		//从原始图像中根据肤色模型提取肤色图像
		cv::bitwise_and(frame,frame,skin,skinmask);
		//水平拼接图片，为了对比
		cv::hconcat(frame, skin, hcon);
		
		//计算肤色检测处理后视频帧率
		time(&end);
		double seconds = difftime (end, start);
		frameCounter++;
		double fps = frameCounter / seconds;
		sprintf(string, "%0.f", fps);      
        std::string fpsString("FPS:");
        fpsString += string;                  
        putText(hcon, fpsString, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); 

		cv::imshow("Skin Detect Example", hcon);
		
		if (cv::waitKey(delay)>=0)
			stop= true;
	}
	// Close the video file.
	// Not required since called by destructor
	capture.release();
    return 0;
}