#include "common.h"
#include "hog.h"
#include "omp.h"
#include "time.h"
#include <cmath>

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

unsigned int segmentDisparity(const Mat &disparity, Mat &output)
{
    output = Mat::zeros(disparity.size(), CV_32SC1);
	Mat tmp=Mat::zeros(disparity.size(), CV_8UC1);
	Mat tmp2=Mat::zeros(disparity.size(), CV_32SC1);
	
  	disparity.convertTo(tmp, CV_8UC1);
	Mat mask=Mat::zeros(disparity.size().height+2,
						disparity.size().width+2,
						CV_8UC1);
	
	Mat mask2 = mask(Rect(1, 1, disparity.size().width, disparity.size().height)); 
	//threshold(tmp, mask2, 0, 255, CV_THRESH_BINARY_INV);
	
	unsigned int k=1;
	
	for(int i=0; i<tmp.rows; i++)
	{
		const unsigned char* buf = tmp.ptr<unsigned char>(i);
		for(int j=0; j<tmp.cols; j++)
		{
			unsigned char d = buf[j];
			if(d>0)
			{
				if(floodFill(tmp,
							 mask,
							 Point(j, i),
							 Scalar(1),
							 NULL,
							 cvScalarAll(0),
							 cvScalarAll(0),
							 8+FLOODFILL_FIXED_RANGE+FLOODFILL_MASK_ONLY)>0)
				{
				  mask2.convertTo(tmp2, CV_32SC1, k);
				  output += tmp2;
				  tmp.setTo(0, mask2);
				  mask.setTo(0);
				  k++;
				}
			}
		}
	}
	return k;
}

//19,230   -   36, 336  panta = dx/dy = (36 -19)/(336-230) ? apropae, ai minus jos :D ok am zis din capos, cum ii la xOy

void fitLineRansac(const std::vector<Point2f> points, Vec4f &line, int iterations, double sigma, double a_max)
{
	int n = points.size();

    if(n<2)
    {
      return;
    }
    
	RNG rng;
	double bestScore = -1.;
	for(int k=0; k<iterations; k++)
	{
		int i1=0, i2=0;
		double dx = 0;
		while(i1==i2)
		{
			i1 = rng(n);
			i2 = rng(n);
		}
		const Point2f& p1 = points[i1];
		const Point2f& p2 = points[i2];
		
		Point2f dp = p2-p1;
		dp *= 1./norm(dp);
		double score = 0;

		if(fabs((double)(dp.x>1.e-5)) && fabs(dp.y/dp.x)<=a_max)
		{
		  for(int i=0; i<n; i++)
		  {
			Point2f v = points[i]-p1;
			double d = v.y*dp.x - v.x*dp.y;
			score += exp(-0.5*d*d/(sigma*sigma));
		  } 
		}
		if(score > bestScore)
		{
          line = Vec4f(dp.x, dp.y, p1.x, p1.y);
          bestScore = score;
		}
	}
}

void detectAndTrack()
{

	// Open image from input file in grayscale
    Mat imgLeft = imread("D:\\Master M2 Grenoble\\Autonomous Robotics\\left_250.png", 0);
	Mat imgRight = imread("D:\\Master M2 Grenoble\\Autonomous Robotics\\right_250.png", 0);
    Mat disparity_img = Mat::zeros(imgLeft.size(), CV_16UC1);
	Mat disparity_img_clone = Mat::zeros(disparity_img.size(), CV_32F);
	Mat display_img (imgLeft.size(), CV_8UC1);
	Mat display_img_clustering(imgLeft.size(), CV_32F);

   clock_t start = clock();
	
   /*computation of disparity data*/
   StereoSGBM  stereo = StereoSGBM(0, 32, 7, 8 * 7 * 7, 32 * 7 * 7, 2, 0, 5, 100, 32, true);
   stereo(imgLeft, imgRight, disparity_img);
   
   disparity_img.convertTo(disparity_img, CV_32F);

   
   float u0 = 258.0, v0 = 156.0, alfa_u = 410.0, alfa_v = 410.0, b = 0.22, z0 = 1.28;

   disparity_img_clone = disparity_img.clone();

   
   /*road/obstacle segmentation in cartesian space*/
#pragma omp parallel for
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
   
   /*road/obstacle segmentation in disparity space*/
   float max_disparity = 32;
   Mat v_disparity = Mat::zeros(disparity_img.rows, max_disparity, CV_32F);

#pragma omp parallel for
   for(int v = 0; v < disparity_img_clone.rows; v++)   
   {
        for(int u = 0; u < disparity_img_clone.cols; u++) 
		{
	  
		  float value = disparity_img_clone.at<float>(v,u)/16.0;

		  if(value > 0.0)
		  {
			  v_disparity.at<float>(v, (int)value) += 1.0;
		  }

		}
	}

   float h0 = 140, p0 = 0.16;
   float theta = - atan((v0 - h0)/alfa_v);


    //apply a threshold
    threshold(v_disparity,v_disparity, 60.0, THRESH_BINARY, THRESH_TOZERO);

	 
	 //add all the remaining points into a vector 
	 vector<Point2f> vec;

	 for(int v = 0; v < v_disparity.rows; v++)   
   {
        for(int u = 0; u < max_disparity; u++) 
		{
			if(v_disparity.at<float>(v,u) != 0.0) 
				vec.push_back(Point2f(v,u));
		}
   }
	 
	 Vec4f line;
	 int iterations = 100;
	 double sigma = 1.0;
	 double a_max = 0.0;

	 //apply Ransac
	 fitLineRansac(vec, line, iterations, sigma, a_max);
	 
     /*clustering in cartesian & disparity space*/
     //apply the erosion operation
     Mat erosion_img = Mat::zeros(disparity_img.size(), CV_32F);
	 Mat dilation_img = Mat::zeros(disparity_img.size(), CV_32F);
	 int erosion_type = MORPH_ELLIPSE;
	 int erosion_size = 2;
	 Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
     erode(disparity_img, erosion_img, element);
	 //apply the dilation operation
	 dilate(erosion_img, dilation_img, element );
	 segmentDisparity(dilation_img, display_img_clustering);
     
     imshow("Left Image", imgLeft);
	 imshow("Right Image", imgRight);
	 
	 //disparity_img.convertTo(display_img, CV_8UC1);
     //imshow("Disparity image", display_img);
	 //dilation_img.convertTo(display_img, CV_8UC1);
	 //imshow("Erosion + dilation image", display_img);
	 display_img_clustering.convertTo(display_img, CV_8UC1);
	 imshow("SegmentDisparity image", display_img);



	 Mat display_vimg(v_disparity.size(), CV_8UC1);
	 v_disparity.convertTo(display_vimg, CV_8UC1);
	 imshow("V-Disparity image", display_vimg);

	 clock_t end = clock();
     float seconds = (float)(end - start) / CLOCKS_PER_SEC;

	 printf("Total time: %.2f",seconds);
	
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