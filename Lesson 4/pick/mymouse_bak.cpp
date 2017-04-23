#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <fstream>
#define PI 3.1415926
using namespace std;

#define max(a,b)            (((a) > (b)) ? (a) : (b))
#define min(a,b)            (((a) < (b)) ? (a) : (b))


IplImage* src=0;
IplImage* dst=0;
IplImage* my=0;
int n=0;
int height, width;
vector<CvPoint> points;


//注意参数是有符号短整型，该函数的作用是使i限定为[a,b]区间内
int bound(short i,short a,short b)
{
 return min(max(i,min(a,b)),max(a,b));
}

CvScalar getInverseColor(CvScalar c)
{
 CvScalar s;
 for(int i=0;i<=2;++i)
 {
  s.val[i]=255-c.val[i];
 }
 return s;
}

void on_mouse( int event, int x, int y, int flags, void* ustc)
{
 CvPoint pt;
 CvPoint tmp_pt = {-1,-1};
 CvFont font;
 cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 4, 4, 0, 1, CV_AA);
 char temp[16], img_name[20];
 CvSize text_size;
 int baseline;

 CvScalar clrPoint=cvScalar(255,0,0,0);
 CvScalar clrText=cvScalar(255, 255, 255, 0);

 if( event == CV_EVENT_MOUSEMOVE )
 {
  cvCopy(dst,src); 

  x=bound(x,0,src->width-1);
  y=bound(y,0,src->height-1);
  pt = cvPoint(x,y);
  cvCircle( src, pt, 20,clrPoint ,CV_FILLED, CV_AA, 0 );

  sprintf(temp,"%d (%d,%d)",n+1,x,y);
  cvGetTextSize(temp,&font,&text_size,&baseline);
  tmp_pt.x = bound(pt.x,0,src->width-text_size.width);
  tmp_pt.y = bound(pt.y,text_size.height+baseline,src->height-1-baseline);
  cvPutText(src,temp, tmp_pt, &font, clrText);

  cvShowImage( "src", src );
 } 
 else if( event == CV_EVENT_LBUTTONDOWN)
 {
  pt = cvPoint(x,y);
  points.push_back(pt); n++;
  cvCircle( src, pt, 2, clrPoint ,CV_FILLED, CV_AA, 0 );

  sprintf(temp,"%d (%d,%d)",n,x,y);
  cvGetTextSize(temp,&font,&text_size,&baseline);
  tmp_pt.x = bound(pt.x,0,src->width-text_size.width);
  tmp_pt.y = bound(pt.y,text_size.height+baseline,src->height-1-baseline);
  cvPutText(src,temp, tmp_pt, &font, clrText);
  cvCopy(src,dst);
  cvShowImage( "src", src );

  //copy roi image
  CvSize msize= cvSize(200, 200);//size of roi
  sprintf(img_name,"x_%d_y_%d.jpg",x,y); //image name
  x = x - msize.width/2;
  y = y - msize.width/2;
  if((x<0||x>src->width)&&(y<0||y>src->height)) 
  {
	  printf("Area Exceed! No Valid ROI\n");
	  return;
  }
  cvSetImageROI(my, cvRect(x, y, msize.width, msize.height));//set the roi of original image
  IplImage* pDest = cvCreateImage(msize, src->depth, src->nChannels);//create dst image
  cvCopy(my,pDest); //copy image
  cvResetImageROI(my);//release ROI
  
  cvSaveImage(img_name ,pDest);//保存目标图像
  cvReleaseImage(&pDest);

 
 } 
}

void convert_coordinate(float &x, float &y)
{
    //int width = (*src).width;
    //int height = (*src).height;

    x = x*(2*PI)/width;
    //y = y*PI/height-(PI/2.0);  //-90~90
	y = y*PI/height;
	////互余
    //if(y > 0.0)
    //{
    //  y = (PI/2.0) - y;
    //}else
    //{
    //  y = -(PI/2.0) - y;
    //}   
}


/*
*argv1: 待标定的图片
*argv2: 标定完成后，保存的图片
*argv3: 保存坐标的文本文件名
*/
int main(int argc ,char *argv[])
{
 src=cvLoadImage(argv[1],1);
 dst=cvCloneImage(src);
 my=cvCloneImage(src);
 height = src->height;
 width = src->width;
 
 cout<<height<<","<<width<<endl;
 cvNamedWindow("src", CV_WINDOW_NORMAL);
 cvSetMouseCallback( "src", on_mouse, 0 );
 
 cvShowImage("src",src);
 cvWaitKey(0); 
 cvDestroyAllWindows();
 cvSaveImage(argv[2], src);
 

 ofstream file(argv[3]);
 if(!file)
 {
  cout << "open file error!";
  return 1;
 }
 vector<CvPoint>::iterator it=points.begin();
 for(;it!=points.end();++it)
 {
  float x = it->x;
  float y = it->y;
  convert_coordinate(x,y);
  file<< x<<','<<y<<endl;
 }
 file<<endl;
 file.close();
 cvReleaseImage(&src);
 cvReleaseImage(&dst);
 cvReleaseImage(&my);
 return 0;
}