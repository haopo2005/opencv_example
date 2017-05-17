#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay();

/** Global variables */
String img_name, eyes_cascade_name;
CascadeClassifier eyes_cascade;
String window_name = "result.jpg";

/** @function main */
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
         "{help h||}"
        "{img_name| image path to be tested|}"
        "{eyes_cascade| trained xml file|}");

    cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
            "You can use Haar or LBP features.\n\n";
    if(!parser.get<string>("help").empty())
	{
		parser.printMessage();
		return 0;
	}
    eyes_cascade_name = parser.get<string>("eyes_cascade");
    cout<<eyes_cascade_name<<endl;
	img_name=parser.get<string>("img_name");
	cout<<img_name<<endl;
   

    //-- 1. Load the cascades
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    detectAndDisplay();
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay()
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat frame = imread(img_name);
    
	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
	imwrite("equalize.jpg", frame_gray);
    //-- Detect faces
    eyes_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, CASCADE_FIND_BIGGEST_OBJECT, Size(30, 14) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }
    //-- Show what you got
    imwrite( window_name, frame );
}
