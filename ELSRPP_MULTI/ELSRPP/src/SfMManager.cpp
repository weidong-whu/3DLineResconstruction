#include "SfMManager.h"
#include"BasicMath.h"
#include <fstream>
template <typename T>
void write2txt(T* mat, int rows, int cols, std::string filename)
{
	std::ofstream writemat(filename, std::ios::out);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			writemat << mat[j] << " ";
		}
		writemat << std::endl;
		mat = mat + cols;
	}
	writemat.close();
}


cv::Mat SfMManager::iCamsRT(int i)
{
	return camsRT[i];
}

void SfMManager::addImSize(int row, int col, int i)
{
	imSizes.at<float>(i, 0) = row;
	imSizes.at<float>(i, 1) = col;
}

void SfMManager::iniImageSize()
{
	imSizes = cv::Mat(cams_number, 2, CV_32FC1);
	imageLineSize.resize(cams_number,0);
}

void SfMManager::setImageLineSize(int lineSize,int i)
{
	imageLineSize[i] = lineSize;
}

void SfMManager::iniCameraSize()
{
	cameraMat = cv::Mat(cams_number, 12, CV_32FC1);
	camera33Trans= cv::Mat(cams_number, 9, CV_32FC1);
}

SfMManager::SfMManager(std::string inputFolder, std::string nvmFile,int NBINS)
{
	input_folder_ = inputFolder;
	nvmFile_ = nvmFile;
	bins = NBINS;
}

std::string SfMManager::inputFolder()
{
	return input_folder_;
}

std::string SfMManager::nvmFile()
{
	return nvmFile_;
}

void SfMManager::addImagePoints(int imageID,cv::Mat pt)
{
	imagePoints[imageID].push_back(pt);
}

void SfMManager::addJunctionLines(int imageID, cv::Mat junc)
{
	junctionLines[imageID].push_back(junc);
}

cv::Mat* SfMManager::iJunctionLines(int imageID)
{
	return &junctionLines[imageID];
}
cv::Mat* SfMManager::iParraLines(int imageID)
{
	return &parraLines[imageID];
}

cv::Mat* SfMManager::iImageLines(int imageID)
{
	return &imageLines[imageID];
}

float* SfMManager::ImageLineSingle(int imageID, int lineID)
{
	return (float*)imageLines[imageID].data + imageLines[imageID].cols * lineID;
}


void SfMManager::initialImagePoints()
{
	imagePoints.resize(cams_number);
	junctionLines.resize(cams_number);
	imageLines.resize(cams_number);
	parraLines.resize(cams_number);
}

void SfMManager::addImageNames(std::string image_name)
{
	imageNames.push_back(image_name);
	cams_number++;
}

std::string SfMManager::iImageNames(int i)
{
	return imageNames[i];
}

void SfMManager::addCamsFocals(float focal_length)
{
	camsFocals.push_back(focal_length);
}


float  SfMManager::iCamsFocals(int i)
{
	return camsFocals[i];
}

void SfMManager::addCamsRT(cv::Mat Rt)
{
	camsRT.push_back(Rt);
}

void SfMManager::addCamsCenter(cv::Mat C)
{
	camsCenters.push_back(C);
}

void SfMManager::add_points_space3D(cv::Mat pos3D)
{
	points3D.push_back(pos3D);
}

void SfMManager::write_points_space3D()
{
	
	write2txt((float*)points3D.data, points3D.rows, points3D.cols, this->inputFolder() + "//output//"+ "spcaepoints.pts");
}

int SfMManager::camsNumber()
{
	return cams_number;
}

float* SfMManager::points3D_ptr()
{
	return (float*)points3D.data;
}


cv::Mat SfMManager::iCameraMat(int i)
{
	return  cameraMat.rowRange(i, i + 1).clone().reshape(0, 3);
}

void SfMManager::writeCmaratxt()
{
	for (int i = 0; i < cameraMat.rows; i++)
	{
		cv::Mat icmaera= cameraMat.rowRange(i, i + 1).clone().reshape(0, 3);
		write2txt((float*)icmaera.data, icmaera.rows, icmaera.cols, input_folder_ + "//output//" + std::to_string(i) + ".P");
	}
	
}

void SfMManager::iImSize(int i, int& row, int& col)
{
	row = imSizes.at<float>(i, 0);
	col = imSizes.at<float>(i, 1);
}

void SfMManager::writeImagePoints()
{
	for (int i = 0; i < imagePoints.size(); i++)
	{
		write2txt((float*)imagePoints[i].data, imagePoints[i].rows, imagePoints[i].cols, input_folder_ + "//output//" + std::to_string(i) + ".pts");
	}
}

void SfMManager::addTrainPrioi(cv::Mat trainPDF,cv::Mat mean_std_, cv::Mat maxv_)
{
	trainPDF.copyTo(train_prioi);
	mean_std_.copyTo(mean_std);
	maxv_.copyTo(max_error);
}



void SfMManager::writeCameraCenters()
{
	for (int i = 0; i < cameraMat.rows; i++)
	{
		cv::Mat icmaera = camsCenters.rowRange(i, i + 1).clone().reshape(0, 1);
		write2txt((float*)icmaera.data, icmaera.rows, icmaera.cols, input_folder_ + "//output//" + std::to_string(i) + ".cen");
	}

	
}



float* SfMManager::iCameraMatPtr(int i)
{
	return (float*)cameraMat.data+cameraMat.cols * i;
}

std::vector<std::string>* SfMManager::allImageNames()
{
	return &imageNames;
}

std::vector<float>* SfMManager::allCamsFocals()
{
	return &camsFocals;
}

std::vector<cv::Mat>* SfMManager::allCameraRTMat()
{
	return &camsRT;
}

cv::Mat* SfMManager::allPoints3D()
{
	return &points3D;
}

void SfMManager::analysis_spacedis()
{
	
	std::vector<std::vector<float>>point_depth(camsNumber());
	float x, y, z, depth;
	int ptid;
	float* CM_ptr;

	for (int i = 0; i < camsNumber(); i++)
	{
		CM_ptr = iCameraMatPtr(i);

		for (int j = 0; j < imagePoints[i].rows; j++)
		{	
			// x = imagePoints[i].at<float>(j, 0);
		    // y = imagePoints[i].at<float>(j, 1);
			ptid = imagePoints[i].at<float>(j, 2) - 1;

			x = points3D.at<float>(ptid, 0);
			y = points3D.at<float>(ptid, 1);
			z = points3D.at<float>(ptid, 2);

			depth = CM_ptr[11]
				  + CM_ptr[8] * x
				  + CM_ptr[9] * y
				  + CM_ptr[10] * z;

			point_depth[i].push_back(depth);
		}
	}

	float pt[2], pt_space[2];
	int imr, imc;

	float ray1[3];
	float ray2[3];
	float ray2_space[3];

	std::vector<float> space_shift;
	std::vector<float> space_shift_space;

	for (int i = 0; i < point_depth.size(); i++)
	{
		if (imagePoints[i].rows==0)
			continue;
		float* M1 = this->iCamera33TransPtr(i);
		ray1[0] = M1[2];
		ray1[1] = M1[5];
		ray1[2] = M1[8];

		std::sort(point_depth[i].begin(), point_depth[i].end());

		float mid_depth = point_depth[i][point_depth[i].size() / 2];

		this->iImSize(i, imr, imc);

		pt[0] = imc / 2.0 + 2.5;
		pt[1] = imr / 2.0 + 2.5;

		pt_space[0] = imc / 2.0 + 0.5;
		pt_space[1] = imr / 2.0 + 0.5;

		solverAxb(M1, pt, ray2);
		solverAxb(M1, pt_space, ray2_space);

		float cos_vec = abs(cos_vec3(ray1, ray2));
		float cos_vec_space = abs(cos_vec3(ray1, ray2_space));

		float space_dis=std::tanf(std::acosf(cos_vec))* mid_depth;
		float space_dis_space = std::tanf(std::acosf(cos_vec_space)) * mid_depth;

		space_shift.push_back(space_dis);
		space_shift_space.push_back(space_dis_space);
	}

	std::sort(space_shift.begin(), space_shift.end());
	std::sort(space_shift_space.begin(), space_shift_space.end());

	
	shift_median = space_shift[space_shift.size() / 2];
	shift_median_min = space_shift_space[space_shift_space.size()/2];

	printf("shift_median %f shift_median_space %f \n", shift_median, shift_median_min);



}

void SfMManager::lineSize(std::vector<int>& linesizevec)
{
	for (int i = 0; i < imageLines.size(); i++)
	{
		linesizevec.push_back(imageLines[i].rows);
	}
}

float* SfMManager::adaptivePDF()
{
	return (float*)train_prioi.data;
}

void SfMManager::getTrainCell(int ind, float& mean, float& std, float& max_err)
{
	mean = mean_std.at<float>(ind, 0);
	std = mean_std.at<float>(ind, 1);
	max_err= max_error.at<float>(ind, 0);

}

void SfMManager::addCameraBySize(int rows, int cols, int i)
{
	cv::Mat K = cv::Mat::zeros(3, 3, CV_32FC1);
	K.at<float>(0, 0) = iCamsFocals(i);
	K.at<float>(1, 1) = iCamsFocals(i);
	K.at<float>(0, 2) = cols / 2.0;
	K.at<float>(1, 2) = rows / 2.0;
	K.at<float>(2, 2) = 1;

	cv::Mat CM_i = K * camsRT[i];
	for (int k = 0; k < 12; k++)
		cameraMat.at<float>(i, k) = ((float*)CM_i.data)[k];

	cv::Mat M = CM_i.colRange(0, 3).clone().t();
	for (int k = 0; k < 9; k++)
		camera33Trans.at<float>(i, k) = ((float*)M.data)[k];





}

cv::Mat* SfMManager::imSizeMat()
{
	return &imSizes;
}

cv::Mat* SfMManager::allCameraMat()
{
	return &cameraMat;
}

int SfMManager::iImageLineSize(int i)
{
	return imageLineSize[i];
}

float* SfMManager::iCamera33TransPtr(int i)
{
	return (float*)(camera33Trans.data) + i* camera33Trans.cols;
}

float* SfMManager::iCameraCenterPtr(int i)
{
	return (float*)(camsCenters.data) + i * 3;
}

cv::Mat* SfMManager::iImagePointsPtr(int i)
{
	return &imagePoints[i];
}

cv::Mat SfMManager::iCameraCenter(int i) {
	return camsCenters.row(i);
}

cv::Mat SfMManager::getImageLines(int imageID) {
	return imageLines[imageID];
}