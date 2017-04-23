#include <opencv/cv.h>    
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>    
#include <math.h>
#include <fstream>
#define PI 3.1415926
typedef  unsigned long uint32;  
typedef  unsigned int  uint16;  
typedef  unsigned char uint8;  
using namespace cv;
using namespace std;
   
IplImage *Dist; 
int height, width;
int pre_x;
uint8    TemplatePixelm10 = 0;  
uint8    TemplatePixelm01 = 0;  
uint8    TemplatePixeln01 = 0;  
uint8    TemplatePixeln10 = 0;  
uint8    TemplatePixelm1n1 = 0;  
uint8    TemplatePixelm1n0 = 0;  
uint8    TemplatePixelm0n1 = 0;  
uint8    TemplatePixelm0n0 = 0;  
  
void CalTemplateDist(IplImage* I,IplImage* Dist, long* Nb)    //计算在模板图对应位置上的距离  
{  
    int i,j;  
    float dis;  
    for ( i=0; i<I->height; i++ )  
    {  
        uint8* ptr = (uint8*)( I->imageData + i*I->widthStep );  
        for ( j=0; j<I->width; j++ )  
        {  
            uint8 Pixel = ptr[j];  
            if(Pixel==0)  
            {  
                dis = 0;  
                (*Nb)++;  
            }  
            else  
            {  
                  
                if( i==0 && j==0 )                     //第一行第一个点  
                {  
                    TemplatePixelm10  = ptr[j+1];  
                    TemplatePixeln10  = *( I->imageData + (i+1)*I->widthStep + j );  
                    TemplatePixelm1n1 = *( I->imageData + (i+1)*I->widthStep + j+1 );  
                    if( TemplatePixelm10==0 || TemplatePixeln10==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm1n1==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else if( i==0 && j>0 && j<(I->width-1) )    //第一行  
                {  
                    TemplatePixelm10  = ptr[j+1];  
                    TemplatePixelm01  = ptr[j-1];  
                    TemplatePixeln10  = *( I->imageData + (i+1)*I->widthStep + j );  
                    TemplatePixelm1n1 = *( I->imageData + (i+1)*I->widthStep + j+1 );  
                    TemplatePixelm0n1 = *( I->imageData + (i+1)*I->widthStep + j-1 );  
                    if( TemplatePixelm10==0 || TemplatePixeln10==0 || TemplatePixelm01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm1n1==0 || TemplatePixelm0n1==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }   
                else if( i==0 && j==(I->width-1) )           //第一行最后一个点  
                {     
                    TemplatePixelm01  = ptr[j-1];  
                    TemplatePixeln10  = *( I->imageData + (i+1)*I->widthStep + j );  
                    TemplatePixelm0n1 = *( I->imageData + (i+1)*I->widthStep + j-1 );  
                    if( TemplatePixeln10==0 || TemplatePixelm01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm0n1==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else if( j==0 && i>0 && i<(I->height-1) )    //第一列  
                {  
                    TemplatePixelm10  = ptr[j+1];  
                    TemplatePixeln01  = *( I->imageData + (i-1)*I->widthStep + j );  
                    TemplatePixeln10  = *( I->imageData + (i+1)*I->widthStep + j );  
                    TemplatePixelm1n1 = *( I->imageData + (i+1)*I->widthStep + j+1 );  
                    TemplatePixelm1n0 = *( I->imageData + (i-1)*I->widthStep + j+1 );  
                    if( TemplatePixelm10==0 || TemplatePixeln10==0 || TemplatePixeln01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm1n1==0 || TemplatePixelm1n0==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else if( i>0 && i<(I->height-1) && j==(I->width-1) )     //最后一列  
                {     
                    TemplatePixelm01  = ptr[j-1];  
                    TemplatePixeln01  = *( I->imageData + (i-1)*I->widthStep + j );  
                    TemplatePixeln10  = *( I->imageData + (i+1)*I->widthStep + j );  
                    TemplatePixelm0n1 = *( I->imageData + (i+1)*I->widthStep + j-1 );  
                    TemplatePixelm0n0 = *( I->imageData + (i-1)*I->widthStep + j-1 );  
                    if( TemplatePixeln10==0 || TemplatePixelm01==0 || TemplatePixeln01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm0n1==0 || TemplatePixelm0n0==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else if( j==0 && i==(I->height-1) )    //最后一行最后一个点  
                {  
                    TemplatePixelm10  = ptr[j+1];  
                    TemplatePixeln01  = *( I->imageData + (i-1)*I->widthStep + j );  
                    TemplatePixelm1n0 = *( I->imageData + (i-1)*I->widthStep + j+1 );  
                    if( TemplatePixelm10==0 || TemplatePixeln01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm1n0==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else if( j>0 && j<(I->width-1) && i==(I->height-1) )    //最后一行  
                {  
                    TemplatePixelm10  = ptr[j+1];  
                    TemplatePixelm01  = ptr[j-1];  
                    TemplatePixeln01  = *( I->imageData + (i-1)*I->widthStep + j );     
                    TemplatePixelm1n0 = *( I->imageData + (i-1)*I->widthStep + j+1 );  
                    TemplatePixelm0n0 = *( I->imageData + (i-1)*I->widthStep + j-1 );  
                    if( TemplatePixelm10==0 || TemplatePixeln01==0 || TemplatePixelm01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm1n0==0 || TemplatePixelm0n0==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else if( j==(I->width-1) && i==(I->height-1) )    //最后一行最后一个点  
                {  
                      
                    TemplatePixelm01  = ptr[j-1];  
                    TemplatePixeln01  = *( I->imageData + (i-1)*I->widthStep + j );                     
                    TemplatePixelm0n0 = *( I->imageData + (i-1)*I->widthStep + j-1 );  
                    if( TemplatePixeln01==0 || TemplatePixelm01==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm0n0==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
                else  
                {  
                    TemplatePixelm10  = ptr[j+1];  
                    TemplatePixelm01  = ptr[j-1];  
                    TemplatePixeln01  = *( I->imageData + (i-1)*I->widthStep + j );  
                    TemplatePixeln10  = *( I->imageData + (i+1)*I->widthStep + j );  
                    TemplatePixelm1n0 = *( I->imageData + (i-1)*I->widthStep + j+1 );  
                    TemplatePixelm0n0 = *( I->imageData + (i-1)*I->widthStep + j-1 );  
                    TemplatePixelm1n1 = *( I->imageData + (i+1)*I->widthStep + j+1 );  
                    TemplatePixelm0n1 = *( I->imageData + (i+1)*I->widthStep + j-1 );  
                    if( TemplatePixeln01==0 || TemplatePixelm01==0 || TemplatePixelm10==0 || TemplatePixeln10==0 )  
                        dis = 0.3;  
                    else if( TemplatePixelm0n0==0 || TemplatePixelm1n0==0 || TemplatePixelm0n1==0 || TemplatePixelm1n1==0 )  
                        dis = 0.7;  
                    else  
                        dis = 1;  
                }  
  
            }  
            *(Dist->imageData + i*Dist->widthStep + j) = (uint8)(dis*255);  
              
        }  
  
    }  
  
}  
void dist_match(IplImage *src_img, IplImage* Dist,double MinMatch, CvPoint* pt, long* Na, long* Nb)  //距离匹配  
{  
    int i,j,m,n;  
    double SigmaGT;  
    uint8 SrcValue;  
    double DistValue;  
    double Match;  
      
  
    for ( i=Dist->height/2; i<(src_img->height - Dist->height/2); i++ )  
    {          
        SigmaGT = 0;  
        *Na = 0;  
        for( m=0; m<Dist->height; m++ )  
        {  
            for( n=0; n<Dist->width; n++ )  
            {  
				SrcValue  = *( src_img->imageData + (i-Dist->height/2+m)*src_img->widthStep + pre_x-Dist->width/2+n );    
                if(SrcValue==0)  
                {  
                    DistValue = (double)*( Dist->imageData +m*Dist->widthStep + n )/255;  
                    SigmaGT += DistValue;  
                    (*Na)++;  
                }  
            }  
        }  
  
        if( (*Na) > (*Nb) )  
        {  
            Match = ( SigmaGT + (*Na) - (*Nb) )/( (*Na) + (*Nb) );  
        }  
        else  
        {  
            Match = ( SigmaGT + (*Nb) - (*Na) )/( (*Na) + (*Nb) );  
        }  
  
        if( Match < MinMatch )  
        {  
            MinMatch = Match;                //距离匹配的最小值就是匹配的位置  
            pt->x = pre_x;  
            pt->y = i;  
        }   
  
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
*argv1:待配准的原始图像
*argv2:模版图像
*argv3:原始图像的canny边缘图
*argv4:模版图像的canny边缘图
*argv5:极线位置
*argv6:匹配位置保存的文本文件名
*argv7:匹配结果图片保存路径
*/
int main(int argc ,char *argv[])  
{  
    IplImage* src_img, *temp_img;                                 //定义变量  
    double MinMatch = 2000;  
    CvPoint MatchPoint = cvPoint(1,1);  
    long Na=0,Nb=0;                                                     //像素0的个数  
    src_img  = cvLoadImage(argv[1],0);                 //读入两幅图  
    temp_img = cvLoadImage(argv[2],0);  
	IplImage *paint = cvLoadImage(argv[1], 1);

	height = src_img->height;
	width = src_img->width;
	pre_x = atoi(argv[5]);

	//提取轮廓
	Mat srcimg = cv::cvarrToMat(src_img);
	Mat tempimg = cv::cvarrToMat(temp_img);
	Mat src_edge, temp_edge;
	
	//blur 灰度图片
    blur(srcimg, srcimg, Size(3,3));
	blur(tempimg, tempimg, Size(3,3));

	// Canny 边缘检测
    Canny(srcimg, src_edge, 10, 30, 3);
	Canny(tempimg, temp_edge, 10, 30, 3);
	imwrite(argv[3],src_edge);
	imwrite(argv[4],temp_edge);

      
    Dist = cvCreateImage(cvGetSize(temp_img),IPL_DEPTH_8U,1);  
    IplImage I = temp_edge;
    CalTemplateDist(&I, Dist, &Nb);  
	IplImage img = src_edge;
    dist_match(&img, Dist, MinMatch, &MatchPoint, &Na, &Nb);  
    printf("%d,%d\n", MatchPoint.x, MatchPoint.y);
    cvRectangle(paint, cvPoint(MatchPoint.x-100,MatchPoint.y-100), cvPoint(MatchPoint.x+100, MatchPoint.y+100), cvScalar(0, 0, 255), 3,8,0);  
    
	ofstream file(argv[6]);
	float x = MatchPoint.x;
	float y = MatchPoint.y;
    convert_coordinate(x, y);
    file<< x<<','<<y<<endl;
	cvSaveImage(argv[7], paint);
	   
    cvReleaseImage(&src_img);    
    cvReleaseImage(&temp_img); 
	cvReleaseImage(&paint);   
  
    return 0;  
}  