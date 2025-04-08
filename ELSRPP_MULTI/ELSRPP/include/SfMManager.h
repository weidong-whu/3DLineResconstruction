#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<string>
#include"IO.h"


class SfMManager
{
	std::vector<std::string> imageNames;
	std::vector<float> camsFocals;
	std::vector<cv::Mat> camsRT;
	
	 
	cv::Mat camera33Trans;
	cv::Mat camsCenters;
	cv::Mat points3D;
	cv::Mat cameraMat;
	cv::Mat imSizes;

	std::string input_folder_;
	std::string nvmFile_;

	int cams_number = 0;

	std::vector<int> imageLineSize;
	std::vector<cv::Mat> imagePoints;
	std::vector<cv::Mat> junctionLines;
	std::vector<cv::Mat> parraLines;
	std::vector<cv::Mat> imageLines;
	//std::vector<cv::Mat> multiPoints;
	
	

public:
	void writeCmaratxt();
	void writeImagePoints();
	void writeCameraCenters();

	float shift_median;
	float shift_median_min;

	void addTrainPrioi(cv::Mat trainPDF, cv::Mat mean_std_, cv::Mat maxv_);

	void addImageNames(std::string image_name);
	std::string iImageNames(int i);
	std::vector<std::string>*allImageNames();

	void addImagePoints(int imageID, cv::Mat pt);
	void initialImagePoints();
	void addJunctionLines(int imageID, cv::Mat junc);

	cv::Mat* iJunctionLines(int imageID);
	cv::Mat* iParraLines(int imageID);
	cv::Mat* iImageLines(int imageID);
	cv::Mat getImageLines(int imageID);
	float* ImageLineSingle(int imageID, int lineID);

	void addCamsFocals(float focal_length);
	float iCamsFocals(int i);
	std::vector<float>* allCamsFocals();

	void addCamsRT(cv::Mat Rt);
	cv::Mat iCamsRT(int i);
	std::vector<cv::Mat>*allCameraRTMat();

	void addCamsCenter(cv::Mat C);
	cv::Mat* allCamsCenters();

	void addCameraBySize(int rows, int cols, int i);
	float* iCameraMatPtr(int i);
	cv::Mat iCameraMat(int i);
	cv::Mat* allCameraMat();
	cv::Mat* iImagePointsPtr(int i);
	cv::Mat iCameraCenter(int i);

	void add_points_space3D(cv::Mat pos3D);
	float* points3D_ptr(); 
	cv::Mat* allPoints3D();

	int camsNumber();

	std::string inputFolder();
	std::string nvmFile();

	SfMManager(std::string inputFolder, std::string nvmFile, int NBINS);

	void iniImageSize();
	void iniCameraSize();
	void setImageLineSize(int lineSize, int i);
	int iImageLineSize(int i);

	void addImSize(int row,int col, int i);
	cv::Mat* imSizeMat();

	void iImSize(int i,int&row,int &col);

	float* iCamera33TransPtr(int i);

	float* iCameraCenterPtr(int i);
	
	void write_points_space3D();

	void analysis_spacedis();

	void lineSize(std::vector<int>&linesizevec);
	
	float interval;
	int bins;

	cv::Mat train_prioi;
	cv::Mat mean_std;
	cv::Mat max_error;

	float* adaptivePDF();

	void getTrainCell(int ind,float& mean,float &std, float& max_err);

	std::vector<cv::Mat> multiPoints;

};



