#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const Size blockSize(16, 16);		
const Size searchSize(48, 48);		

const int startingFrame = 1;

const string backgroundWindowName = "Background";
const string motionWindowName = "Video";
const int sumCount = 50;		
const double thresh = 25;

Mat frame2YUV(Mat);
Mat izostravanje(Mat);
void HarrisCornerDet(Mat &src_gray, Mat &HarrisC, int thresh);
void detect_Harris(Mat &GxABS, Mat &GyABS, Mat &R);
static void drawArrows(Mat& img, Point& prevPts, Point& nextPts, Scalar line_color);

int main(int argc, char** argv) {

	VideoCapture videjo("kazoo.mp4");

	VideoWriter outputVideo;

	int numberofframes = static_cast<int>(videjo.get(CV_CAP_PROP_FRAME_COUNT));

	if (!videjo.isOpened()) {
		cout << "Ne moze da se otvori video" << endl;
	}
	waitKey(3000);

	Size S = Size((int)videjo.get(CV_CAP_PROP_FRAME_WIDTH), (int)videjo.get(CV_CAP_PROP_FRAME_HEIGHT));

	int xCount = S.width / blockSize.width;		
	int yCount = S.height / blockSize.height;

	int fps = videjo.get(CV_CAP_PROP_FPS);
	int ex = -1;

	videjo.set(CV_CAP_PROP_POS_FRAMES, startingFrame);		

	Mat previousFrame;


	outputVideo.open("videjo.avi", ex, fps, S, true);

	int t=0;
	deque<Mat> previousFrames;						
	Mat sum = Mat::zeros(S, CV_32F);

	bool paused = false;
	while (true) 
	{
			Mat frame;
			videjo >> frame;

			if(frame.empty()) {
				break;
			}

			Mat YUV = frame2YUV(frame);
			Mat Y[3];
			split(YUV, Y);
			Mat frameY=izostravanje(Y[0]);
			frameY.convertTo(frameY,CV_32F);

			outputVideo << frameY;

			Mat displayFrame;
			if (!sum.empty())
			{
				
				Mat background = sum / sumCount;
				normalize(background, displayFrame, 0, 1, CV_MINMAX);
				imshow(backgroundWindowName, displayFrame);
				absdiff(background, frameY, displayFrame);
				threshold(displayFrame, displayFrame, thresh, NULL, THRESH_TOZERO);
				normalize(displayFrame, displayFrame, 0, 1, CV_MINMAX);
			
			}

			sum += frameY;
			previousFrames.push_back(frameY);

			if (previousFrames.size() > sumCount)
			{
				Mat firstFrame = previousFrames.front();
				previousFrames.pop_front();
				sum -= firstFrame;
			}

			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, 1));
			Mat ImgErozija;
			erode(displayFrame, ImgErozija, element);
			Mat ImgED;
			dilate(ImgErozija, ImgED, element);
			Mat Harris;
			HarrisCornerDet(ImgED,Harris,150);
			imshow(motionWindowName, Harris);

			displayFrame=Harris;

			if (!previousFrame.empty())		
			{
				cvtColor(displayFrame, displayFrame, CV_RGB2GRAY);
				cvtColor(displayFrame, displayFrame, CV_GRAY2RGB);

				for (int itBlockX = 0; itBlockX < xCount; itBlockX++)		
					for (int itBlockY = 0; itBlockY < yCount; itBlockY++)
					{
						Point blockPos(itBlockX * blockSize.width, itBlockY * blockSize.height);

						Point searchPos(blockPos.x - (searchSize.width - blockSize.width) / 2, blockPos.y - (searchSize.height - blockSize.height) / 2);

						Rect blockRect(blockPos, blockSize);		
						Mat block = previousFrame(blockRect);	

						Point bestPos;		        
						float bestResult = 100000; 

						int XDimSearch = searchSize.width - blockSize.width;
						int YDimSearch = searchSize.height - blockSize.height;



 						for (int itSearchX = 0; itSearchX < XDimSearch; itSearchX += 4)		
							for (int itSearchY = 0; itSearchY < YDimSearch; itSearchY += 4)
							{
								Point kernelPos(searchPos.x + itSearchX, searchPos.y + itSearchY);

								if (kernelPos.x < 0 || kernelPos.x + blockSize.width > S.width) continue;		
								if (kernelPos.y < 0 || kernelPos.y + blockSize.height > S.height) continue;

								Rect kernelRect(kernelPos, blockSize);		
								Mat kernel = frame(kernelRect);											

								Mat diff;							
								absdiff(block, kernel, diff);		
								Scalar result = mean(diff);			

								if (result[0] < bestResult)			
								{
									bestResult = result[0];
									bestPos = kernelPos;
								}
							}

						Rect bestRect(bestPos.x, bestPos.y, blockSize.width, blockSize.height);

							if(blockPos!=bestPos)  //ukoliko ocete i tackice sa strane samo udrite komentare
								drawArrows(displayFrame,blockPos, bestPos,  CV_RGB(0,20,255));//teo sam tirkiznu plavu boju tacka
					}

			}
			namedWindow("Block", CV_WINDOW_AUTOSIZE);
			imshow("Block", displayFrame);		

			frame.copyTo(previousFrame);

		
		if(waitKey(30)>=0)
			break;
	}
	waitKey(0);
	return 0;
}

Mat frame2YUV(Mat inImg) {
	int width = inImg.cols;
	int height = inImg.rows;

	Mat YUVImage(height, width, CV_8UC3, Scalar(0, 0, 0));

	double Y, U, V, R, G, B;

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			R = inImg.at<Vec3b>(j, i).val[0];
			G = inImg.at<Vec3b>(j, i).val[1];
			B = inImg.at<Vec3b>(j, i).val[2];

			Y = 0.299*R + 0.587*G + 0.114*B;
			U = 0.436*(B - Y) / (1 - 0.114);
			V = 0.615*(R - Y) / (1 - 0.299);

			YUVImage.at<Vec3b>(j, i).val[0] = Y;
			YUVImage.at<Vec3b>(j, i).val[1] = U;
			YUVImage.at<Vec3b>(j, i).val[2] = V;
		}
	}
	return YUVImage;
}

Mat izostravanje(Mat img) {

	Mat Gaus;
	GaussianBlur(img, Gaus, Size(3, 3), 1, 1);

	Size kernelSize(3, 3);
	float valLaplas[] = {
				0.,-1.,0.,
				-1.,4.,-1.,
				0.,-1.,0. };
	Mat kernelLaplas(kernelSize, CV_32FC1, valLaplas);

	Mat Laplas;
	filter2D(img, Laplas, -1, kernelLaplas);
	convertScaleAbs(Laplas, Laplas);

	Mat Yizostren;
	addWeighted(Gaus, 1.5, Laplas, 0.8, 0, Yizostren);

	return Yizostren;
}

void HarrisCornerDet(Mat &src_gray, Mat &HarrisC, int thresh){

		Mat Gx, Gy, GxABS, GyABS, R;

	Point anchor = Point(-1, -1);
	int delta = 0;
	int ddepth = -1;

	Size kernelSizeX(3, 3);
	float valuesX[] = {
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1 };
	Mat kernelX(kernelSizeX, CV_32FC1, valuesX);

	Size kernelSizeY(3, 3);
	float valuesY[] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1 };
	Mat kernelY(kernelSizeY, CV_32FC1, valuesY);

	int width = src_gray.cols;
	int height = src_gray.rows;

	R = Mat::zeros(height / 8, width / 8, CV_32FC1);

	filter2D(src_gray, Gx, ddepth, kernelX, anchor, delta, BORDER_DEFAULT);
	convertScaleAbs(Gx, GxABS);

	filter2D(src_gray, Gy, ddepth, kernelY, anchor, delta, BORDER_DEFAULT);
	convertScaleAbs(Gy, GyABS);

	detect_Harris(GxABS, GyABS, R);

	Mat Harris = src_gray.clone();
	Mat R_NORM, R_NORM_SCALED;

	cvtColor(Harris, HarrisC, CV_GRAY2RGB);

	normalize(R, R_NORM, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(R_NORM, R_NORM_SCALED);

	for (int j = 0; j < height - 8; j += 8)
	{
		for (int i = 0; i < width - 8; i += 8)
		{
			if ((int)R_NORM.at<float>(j / 8, i / 8) > thresh)
			{
				circle(HarrisC, Point(i, j), 5, Scalar(255, 255, 255), 2, 8, 0);
			}
		}
	}

}

void detect_Harris(Mat &GxABS, Mat &GyABS, Mat &R)
{
	int w = GxABS.cols;
	int h = GxABS.rows;
	int SumX, SumY, SumXY;
	double k = 0.04;
	float val;

	for (int i = 0; i < w - 8; i += 8)
		for (int j = 0; j < h - 8; j += 8)
		{
		SumX = 0;
		SumY = 0;
		SumXY = 0;

		for (int m = 0; m < 8; m++)
			for (int n = 0; n < 8; n++)
			{
			SumX += GxABS.at<uchar>(j + n, i + m) * GxABS.at<uchar>(j + n, i + m);
			SumY += GyABS.at<uchar>(j + n, i + m) * GyABS.at<uchar>(j + n, i + m);
			SumXY += GxABS.at<uchar>(j + n, i + m) * GyABS.at<uchar>(j + n, i + m);
			}

		val = (((SumX * SumY) - (SumXY * SumXY)) - k*((SumX + SumY)*(SumX + SumY)));
		R.at<float>(j / 8, i / 8) = val;
		}

}

static void drawArrows(Mat& img, Point& prevPts, Point& nextPts, Scalar line_color)
{

            int line_thickness = 2;

            Point p = prevPts;
            Point q = nextPts;

            double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

            double hypotenuse = sqrt( ( (double)p.y - (double)q.y)*( (double)p.y - (double)q.y) + ( (double)p.x - (double)q.x)*( (double)p.x - (double)q.x));

            q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
            q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

            line(img, p, q, line_color, line_thickness);
            p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
            line(img, p, q, line_color, line_thickness);

            p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
            line(img, p, q, line_color, line_thickness);

}