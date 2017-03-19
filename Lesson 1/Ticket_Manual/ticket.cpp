#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

struct userdata{
    Mat im;
    vector<Point2f> points;
};


void mouseHandler(int event, int x, int y, int flags, void* data_ptr)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        userdata *data = ((userdata *) data_ptr);
        circle(data->im, Point(x,y),3,Scalar(0,255,255), 5, CV_AA);
        imshow("Image", data->im);
        if (data->points.size() < 4)
        {
            data->points.push_back(Point2f(x,y));
        }
    }
    
}

int main( int argc, char** argv)
{
    // Destination image
    Mat im_dst = imread(argv[1]);
    
    // Set data for mouse handler
    Mat im_temp = im_dst.clone();
    userdata data;
    data.im = im_temp;

    namedWindow("Image", CV_WINDOW_NORMAL);
    //show the image
    imshow("Image", im_temp);
    
    cout << "Click on four corners of a billboard and then press ENTER" << endl;
    //set the callback function for any mouse event
    setMouseCallback("Image", mouseHandler, &data);
    waitKey(0);
    
	float x_min = data.points[0].x; 
	float x_max = data.points[0].x;
	float y_min = data.points[0].y; 
	float y_max = data.points[0].y;
	
	for(int i = 0; i < 4;i++)
	{
		if(x_min > data.points[i].x) x_min = data.points[i].x;
		if(x_max < data.points[i].x) x_max = data.points[i].x;
		if(y_min > data.points[i].y) y_min = data.points[i].y;
		if(y_max < data.points[i].y) y_max = data.points[i].y;
	}
	float width = x_max - x_min;
    float height = y_max - y_min;	
	// Create a vector of points.
    vector<Point2f> pts_src;
    pts_src.push_back(Point2f(0,0));
    pts_src.push_back(Point2f(width - 1, 0));
    pts_src.push_back(Point2f(width - 1, height -1));
    pts_src.push_back(Point2f(0, height - 1 ));
	
    // Calculate Homography between source and destination points
    Mat h = getPerspectiveTransform(data.points, pts_src);
    cout<<"pts_src: "<<pts_src<<endl;
	cout<<"data.points: "<<data.points<<endl;
	cout<<"h: "<<h<<endl;
    // Warp source image
	Mat dst_image;
    warpPerspective(im_temp, dst_image, h, cv::Size(width, height));
    
    // Display image.
	Mat dst_grey, dst_binary;
	cvtColor(dst_image, dst_grey, CV_BGR2GRAY);
	adaptiveThreshold(dst_grey, dst_binary, 255,
                        CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
    imwrite("result.jpg", dst_binary);
    imshow("Image", dst_binary);
    waitKey(0);

    return 0;
}