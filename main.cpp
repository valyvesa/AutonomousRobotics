#include "common.h"
#include "hog.h"

using namespace cv;
using namespace std;

void testOpenImage()
{
	char fname[255];

	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char fname[255];
	
	FileGetter fg("Images","bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		
		imshow("image",src);
		waitKey();
	}
}

void testNegativeImage()
{
	char fname[255];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255-val;
				dst.at<uchar>(i,j) = neg;
			}
		}
		
		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[255];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testResize()
{
	char fname[255];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);

		Mat dst1,dst2;

		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[255];
	while(openFileDlg(fname))
	{
		Mat src,dst;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);

		Canny(src,dst,40,100,3);
		
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testPedestrianDetection()
{
	char fname[255];

	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);

		setupDetector();
		onPDFStart(src, src);

		imshow("image", src);
		waitKey();
	}
}

void detectAndTrack()
{

	// Open image from input file in grayscale
    Mat imgLeft = imread("D:\\Master M2 Grenoble\\Autonomous Robotics\\left_250.png", 0);
	Mat imgRight = imread("D:\\Master M2 Grenoble\\Autonomous Robotics\\right_250.png", 0);
    Mat disparity_img (imgLeft.size(), CV_16UC1);
	Mat display_img (imgLeft.size(), CV_8UC1);
	

   StereoSGBM  stereo = StereoSGBM(0, 32, 7, 8 * 7 * 7, 32 * 7 * 7, 2, 0, 5, 100, 32, true);
   stereo(imgLeft, imgRight, disparity_img);
   
   disparity_img.convertTo(disparity_img, CV_32F);

   
   float u0 = 258.0, v0 = 156.0, alfa_u = 410.0, alfa_v = 410.0, b = 0.22, z0 = 1.28;

   
   for(int v = 0; v < disparity_img.rows; v++) 
	  {

        for(int u = 0; u < disparity_img.cols; u++) 
		{
	  
		  float value = disparity_img.at<float>(v,u);
          float d = value/16.0;
		  float x = (((u - u0) * b)/d) - (b/2.0);
		  float y = (alfa_u * b)/d;
		  float z = z0 - ((v - v0) * alfa_u * b)/(alfa_v * d);

		  if(z < 0.2 || z > 2.5) disparity_img.at<float>(v,u) = 0.0;

        }
	
      }
   

   /*
   float max_disparity = 0.0;
   for(int v = 0; v < disparity_img.rows; v++)   
   {

        for(int u = 0; u < disparity_img.cols; u++) 
		{
	  
		  float value = disparity_img.at<float>(v,u);

		  if(value > max_disparity) max_disparity = value;
		}
	}
	
   
   Mat v_disparity(32, disparity_img.rows, CV_32F);
	//compute v-disparity
	for(int v = 0; v < disparity_img.rows; v++)
	{
		for(int u = 0; u < disparity_img.cols; u++)
		{
			if(disparity_img.at<float>(v,u) > 0.0)
			{
				v_disparity.at<float>((int) disparity_img.at<float>(v,u), v) +=1;
			}
		}
	}
	*/

     imshow("Left Image", imgLeft);
	 imshow("Right Image", imgRight);
	 
	 //disparity_img.convertTo(display_img, CV_8UC1);
	 v_disparity.convertTo(display_img, CV_8UC1);
     imshow("Output image", display_img);

	 cout<<CV_MAJOR_VERSION<<" "<<CV_MINOR_VERSION << " " << CV_SUBMINOR_VERSION;

	 waitKey();
	
}

int main()
{
	setupDetector();


	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open images from folder\n");
		printf(" 3 - Negative imagine\n");
		printf(" 4 - Color->Grayscale\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Pedestrian detection\n");
		printf(" 8 - Detect & Track moving objects\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);

		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testColor2Gray();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testPedestrianDetection();
				break;
			case 8:
				detectAndTrack();
				break;
		}
	}
	while (op!=0);

	return 0;
}