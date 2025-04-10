#include "SingleImage.h"
#include <thread>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "line_detector/lsd.hpp"
#include "BasicMath.h"
#include <fstream>
#include "Parameters.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include"IO.h"

#include "line_detector/ag3line/ag3line_go.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include<vector>

#include"line_detector/EDLineDetector.h"

using namespace cv;
using namespace std;

static  void ang_grad_simple(const Mat& im, Mat& grad, Mat& ang)
{
	//calculate the grad and angle data
	//define the x and y direction kernal
	int rsize = im.rows, colsize = im.cols;
	float dx[] = { -1,1,-1,1 };
	float dy[] = { -1,-1,1,1 };
	Mat mdx = Mat(2, 2, CV_32FC1, dx);
	Mat mdy = Mat(2, 2, CV_32FC1, dy);

	Mat gx, gy;
	filter2D(im, gx, im.depth(), mdx);
	filter2D(im, gy, im.depth(), mdy);

	grad = Mat::zeros(rsize, colsize, CV_32FC1);
	ang = Mat::zeros(rsize, colsize, CV_32FC1);

	//access each pixel to calculate the angle
	float* ptrdx = (float*)gx.data;
	float* ptrdy = (float*)gy.data;
	float* ptrgrad = (float*)grad.data;
	float* ptrang = (float*)ang.data;
	float vdx, vdy, vgrad;
	for (int i = 0; i < rsize; i++)
	{
		for (int j = 0; j < colsize; j++)
		{
			vdx = *(ptrdx++);
			vdy = *(ptrdy++);
			vgrad = vdx * vdx + vdy * vdy;


			*(ptrgrad++) = std::sqrt(vgrad);

			//*(ptrang++)=cv::fastAtan2(vdy,vdx);
			*(ptrang++) = std::atan2(vdy, vdx);
			//cout<<std::atan2(vdy,vdx)<<" "<<atan2approx(vdy,vdx)<<endl;;
			//getchar();
		}
	}

}


float clockwise_angle(float* v1, float* v2)
{
	float x1, y1, x2, y2, dot, det, theta;
	x1 = v1[0];
	y1 = v1[1];

	x2 = v2[0];
	y2 = v2[1];

	dot = x1 * x2 + y1 * y2;
	det = x1 * y2 - y1 * x2;

	theta = std::atan2(det, dot);

	if (theta < 0)
		theta = 2 * 3.14 + theta;

	return theta;
}
	
	

bool line_direction_change(int x1,
	int y1,
	int const x2,
	int const y2,
	int imr, int imc,
	Mat direction) {
	
	float v1[2], v2[2];
	
	std::vector<int> xx;
	std::vector<int> yy;
	Bresenham(x1, y1, x2, y2, xx, yy);

	std::vector<float> node_direction(xx.size());
	float line_grad_direction = 0.0;

	float sumdx = 0, sumdy = 0;
	for (int i = 0; i < xx.size(); i++) {

		if (xx[i]<0 || xx[i]>=imc || yy[i]<0 || yy[i]>=imr)
			continue;
		sumdx += cos(direction.at<float>(yy[i], xx[i]));
		sumdy += sin(direction.at<float>(yy[i], xx[i]));
	}

	v1[0] = sumdx;
	v1[1] = sumdy;

	v2[0] = x2 - x1;
	v2[1] = y2 - y1;

	float angdiff=clockwise_angle(v1, v2);


	if (angdiff< 3.14) {
		return true;
	}
	return false;
}


void allignLineSegments(cv::Mat* lines, cv::Mat& img_float)
{

	Mat direction, grad;
	//get_img_direction(img_float, direction);
	ang_grad_simple(img_float, grad, direction);
	int true_count = 0;
	float x1, y1, x2, y2;
	for (int i = 0; i < lines->rows; i++) {

		x1 = lines->at<float>(i, 0);
		y1 = lines->at<float>(i, 1);

		x2 = lines->at<float>(i, 2);
		y2 = lines->at<float>(i, 3);

		bool change = line_direction_change(x1, y1, x2, y2, img_float.rows, img_float.cols, direction);
		if (!change) {

			lines->at<float>(i, 0) = x2;
			lines->at<float>(i, 1) = y2;

			lines->at<float>(i, 2) = x1;
			lines->at<float>(i, 3) = y1;

			true_count += 1;
		}

	}
	//cout << true_count;
}




//using namespace cv::ximgproc;
void points2KDtree(cv::Mat inter_lines_Mf, cv::Mat* inter_knn_Mi, cv::Mat lines_Mf,
	cv::Mat* lines_knn_Mi_1, cv::Mat* lines_knn_Mi_2,
	int SUPPORT_POINT_NUM1, int SUPPORT_POINT_NUM2, SfMManager* sfm, int imageID, float imr, float imc)
{

	std::string pt_file_adr;
	std::ifstream pt_file;
	std::string pt_line;

	cv::Mat pt2_Mf;
	cv::Mat pt2_idpt3__Mf;

	cv::Mat pt2 = *(sfm->iImagePointsPtr(imageID));

	if (pt2.rows < SUPPORT_POINT_NUM2 || pt2.rows < SUPPORT_POINT_NUM1)
	{
		printf("too few points: %d\n", pt2.rows);
		getchar();
	}



	pt2_Mf = pt2.colRange(0, 2).clone();
	pt2_idpt3__Mf = pt2.colRange(2, 3).clone();

	// Allocate memory for computed k-NN neighbors
	int query_nb = pt2_Mf.rows;
	int k1 = SUPPORT_POINT_NUM1;

	float imr_2 = imr / 2.0;
	float imc_2 = imc / 2.0;

	for (int i = 0; i < pt2_Mf.rows; i++)
	{
		pt2_Mf.at<float>(i, 0) = pt2_Mf.at<float>(i, 0) + imc_2;
		pt2_Mf.at<float>(i, 1) = pt2_Mf.at<float>(i, 1) + imr_2;
	}


	// construct kdtree
	cv::flann::Index flannIndex(pt2_Mf, cv::flann::KDTreeIndexParams());

	// intersection KD tree
	cv::Mat inter_2_pt3index_M;
	cv::Mat inter_2_pt3index_dist;
	cv::Mat inter_pts = inter_lines_Mf.colRange(2, 4).clone();
	//printf("\n %d %d %d in.............\n", inter_pts.rows, pt2_Mf.rows, SUPPORT_POINT_NUM1);
	flannIndex.knnSearch(inter_pts, inter_2_pt3index_M,
		inter_2_pt3index_dist, SUPPORT_POINT_NUM1, cv::flann::SearchParams
		());

	//printf("\n in.............\n");
	*inter_knn_Mi = cv::Mat(inter_pts.rows, SUPPORT_POINT_NUM1, CV_32SC1);

	int* inter_knn_Mi_ptr = (int*)(*inter_knn_Mi).data;
	int* inter_2_pt3index_M_ptr = (int*)inter_2_pt3index_M.data;

	int ind_inter = 0;
	for (int i = 0; i < inter_pts.rows; i++)
	{
		for (int j = 0; j < SUPPORT_POINT_NUM1; j++)
		{

			inter_knn_Mi_ptr[ind_inter] =
				pt2_idpt3__Mf.at<float>(inter_2_pt3index_M_ptr[ind_inter], 0) - 1;
			//-1 is aligned to main_vsfm.m file
			ind_inter++;
		}

	}


	// line segment KD tree
	cv::Mat line_2_pt3index_1;
	cv::Mat line_2_pt3index_dist1;
	cv::Mat line_pts1 = lines_Mf.colRange(0, 2).clone();
	flannIndex.knnSearch(line_pts1, line_2_pt3index_1,
		line_2_pt3index_dist1, SUPPORT_POINT_NUM2, cv::flann::SearchParams());

	cv::Mat line_2_pt3index_2;
	cv::Mat line_2_pt3index_dist2;
	cv::Mat line_pts2 = lines_Mf.colRange(2, 4).clone();
	flannIndex.knnSearch(line_pts2, line_2_pt3index_2,
		line_2_pt3index_dist2, SUPPORT_POINT_NUM2, cv::flann::SearchParams());

	//store
	*lines_knn_Mi_1 = cv::Mat(lines_Mf.rows, SUPPORT_POINT_NUM2, CV_32SC1);
	*lines_knn_Mi_2 = cv::Mat(lines_Mf.rows, SUPPORT_POINT_NUM2, CV_32SC1);

	int* lines_knn_Mi_1_ptr = (int*)lines_knn_Mi_1->data;
	int* lines_knn_Mi_2_ptr = (int*)lines_knn_Mi_2->data;

	int* line_2_pt3index_1_ptr = (int*)line_2_pt3index_1.data;
	int* line_2_pt3index_2_ptr = (int*)line_2_pt3index_2.data;

	int  ind_ptr = 0;
	for (int i = 0; i < lines_Mf.rows; i++)
		for (int j = 0; j < SUPPORT_POINT_NUM2; j++)
		{

			lines_knn_Mi_1_ptr[ind_ptr] =
				pt2_idpt3__Mf.at<float>(line_2_pt3index_1_ptr[ind_ptr], 0) - 1;

			lines_knn_Mi_2_ptr[ind_ptr] =
				pt2_idpt3__Mf.at<float>(line_2_pt3index_2_ptr[ind_ptr], 0) - 1;
			//-1 is aligned to vsfm.m file
			ind_ptr = ind_ptr + 1;
		}

	//printf("\n out.............\n");
}


bool points_knn(cv::Mat queryPoints, cv::Mat keyPoints, cv::Mat keyPoints_ind, int k, cv::Mat* queryRes)
{


	cv::Mat indices, dists;
	// construct kdtree
	cv::flann::Index flannIndex(keyPoints, cv::flann::KDTreeIndexParams());

	flannIndex.knnSearch(queryPoints, indices, dists, k, cv::flann::SearchParams());

	*queryRes = cv::Mat(queryPoints.rows, k, CV_32SC1);
	int* queryRes_ptr = (int*)(*queryRes).data;

	int ind_inter = 0;
	int subInd = 0;


	for (int i = 0; i < queryPoints.rows; i++)
	{
		for (int j = 0; j < k; j++)
		{
			subInd = indices.at<int>(i, j);
			//printf("%d\n",subInd);
			queryRes_ptr[ind_inter]
				= keyPoints_ind.at<float>(subInd, 0) - 1;
			ind_inter++;
		}
	}
	return true;
}



void readLsdLines(cv::Mat& lines_Mf, std::string line_file_adr,int maxLineNum)
{
	
	std::ifstream line_file;
	line_file.open(line_file_adr.c_str());
	std::string line_line;

	float cx, cy, x1, y1, x2, y2, length;

	cv::Mat line_Mf_;
	cv::Mat per_row(1, 7, CV_32FC1);
	float* per_row_p = (float*)per_row.data;
	while (std::getline(line_file, line_line))
	{
		std::istringstream iss_imidx(line_line);

		//std::cout << line_line << std::endl;

		iss_imidx >> x1;
		
		iss_imidx >> y1;

		
		iss_imidx >> x2;  
		iss_imidx >> y2;

		//printf("%f %f %f %f\n", x1, y1, x2, y2);
		//getchar();

		length = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

		cx = (x1 + x2) / 2;
		cy = (y1 + y2) / 2;

		per_row_p[0] = x1;
		per_row_p[1] = y1;
		per_row_p[2] = x2;
		per_row_p[3] = y2;
		per_row_p[4] = cx;
		per_row_p[5] = cy;
		per_row_p[6] = length;

		line_Mf_.push_back(per_row);
	}

	line_file.close();

	

	if (line_Mf_.rows < maxLineNum)
	{
		line_Mf_.copyTo(lines_Mf);
		std::cout << "line size " << lines_Mf.rows << std::endl;
		return;
	}

	cv::Mat order;
	cv::sortIdx(line_Mf_.colRange(6, 7), order, cv::SORT_DESCENDING + cv::SORT_EVERY_COLUMN);

	for (int i = 0; i < maxLineNum; i++)
		lines_Mf.push_back(line_Mf_.row(order.at<int>(i, 0)));

	
	std::cout << "line size " << lines_Mf.rows << std::endl;

	
}


void maxMinDepth(int support_num, int lp_index, float* spacepoints,
	int* inter_knn, float* CM,
	float& mindepth, float& maxdepth)
{
	mindepth = 99999;
	maxdepth = 0;
	float w;
	int ind, indpt3;
	ind = lp_index * support_num;

	//std::cout << lp_index << " ";
	for (int i = 0; i < support_num; i++)
	{

		//std::cout << inter_knn[ind + i] << " ";
		indpt3 = inter_knn[ind + i] * 3;

		w = CM[11]
			+ CM[8] * spacepoints[indpt3]
			+ CM[9] * spacepoints[indpt3 + 1]
			+ CM[10] * spacepoints[indpt3 + 2];

		if (maxdepth < w)
			maxdepth = w;

		if (mindepth > w)
			mindepth = w;

	}

}

void maxMinPt(int feature_size, int support_num, float* spacepoints,
	int* inter_knn, float* CM, float* out_range)
{
	float mindepth, maxdepth;
	float w;
	int  ind, indpt3;
	for (int i = 0; i < feature_size; i++)
	{
		mindepth = 99999;
		maxdepth = 0;

		ind = i * support_num;

		//std::cout << lp_index << " ";
		for (int j = 0; j < support_num; j++)
		{
			//std::cout << inter_knn[ind + i] << " ";
			indpt3 = inter_knn[ind + j] * 3;

			w = CM[11]
				+ CM[8] * spacepoints[indpt3]
				+ CM[9] * spacepoints[indpt3 + 1]
				+ CM[10] * spacepoints[indpt3 + 2];

			if (maxdepth < w)
				maxdepth = w;

			if (mindepth > w)
				mindepth = w;
		}

		out_range[i * 2] = mindepth;
		out_range[i * 2 + 1] = maxdepth;

	}
}

float siftPixelAng(cv::Mat CM, float imr1, float imc1, float shift_p)
{
	float M[] = { CM.at<float>(0,0),CM.at<float>(0,1),CM.at<float>(0,2),
				  CM.at<float>(1,0),CM.at<float>(1,1),CM.at<float>(1,2),
				  CM.at<float>(2,0),CM.at<float>(2,1),CM.at<float>(2,2) };

	imr1 = imr1 / 2;;
	imc1 = imc1 / 2;;

	float v1[3];
	float v2[3];

	float size_v1[] = { imc1,imr1,1 };
	float size_v2[] = { size_v1[0] + shift_p,size_v1[1],1 };

	M_divide_b(M, size_v1, v1);
	M_divide_b(M, size_v2, v2);

	float cos_v = cos_vec3(v1, v2);
	return  1 - cos_v * cos_v;
}


void dect_ag3_lines(cv::Mat img, cv::Mat& lines_Mf_, float scale)
{
	std::vector<lineag>lines;

	ag3line(img, lines, true);
	int line_num = lines.size();
	// note -1 is minored for iteration
	// add the lines to opencv Mat
	cv::Mat lines_Mf = cv::Mat(line_num, 7, CV_32FC1);
	float* lines_Mf_ptr = (float*)lines_Mf.data - 1;
	// note -1 is minored for iteration
	float x1, y1, x2, y2, cx, cy, length;
	for (int j = 0; j < line_num; j++)
	{
		x1 = lines[j].x1 / scale;
		y1 = lines[j].y1 / scale;
		x2 = lines[j].x2 / scale;
		y2 = lines[j].y2 / scale;
		cx = (x2 + x1) / 2;
		cy = (y2 + y1) / 2;

		length = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
		length = std::sqrt(length);

		*(++lines_Mf_ptr) = x1;
		*(++lines_Mf_ptr) = y1;
		*(++lines_Mf_ptr) = x2;
		*(++lines_Mf_ptr) = y2;
		*(++lines_Mf_ptr) = cx;
		*(++lines_Mf_ptr) = cy;
		*(++lines_Mf_ptr) = length;
	}
	lines_Mf.copyTo(lines_Mf_);
}

void dect_lsd_lines(cv::Mat img, cv::Mat* lines_Mf, float scale)
{
	cv::Mat img_double;
	img.convertTo(img_double, CV_32FC1);

	int line_num;
	float* img_ptr = (float*)img_double.data;
	float* line_lsd_ptr = lsd_scale(&line_num, img_ptr, img_double.cols, img_double.rows, 1.0) - 1;
	// note -1 is minored for iteration

	// add the lines to opencv Mat
	*lines_Mf = cv::Mat(line_num, 7, CV_32FC1);
	float* lines_Mf_ptr = (float*)lines_Mf->data - 1;
	// note -1 is minored for iteration
	float x1, y1, x2, y2, cx, cy, length;
	for (int j = 0; j < line_num; j++)
	{
		x1 = *(++line_lsd_ptr) / scale;
		y1 = *(++line_lsd_ptr) / scale;
		x2 = *(++line_lsd_ptr) / scale;
		y2 = *(++line_lsd_ptr) / scale;
		cx = *(++line_lsd_ptr) / scale;
		cy = *(++line_lsd_ptr) / scale;
		++line_lsd_ptr;

		length = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
		length = std::sqrt(length);

		*(++lines_Mf_ptr) = x1;
		*(++lines_Mf_ptr) = y1;
		*(++lines_Mf_ptr) = x2;
		*(++lines_Mf_ptr) = y2;
		*(++lines_Mf_ptr) = cx;
		*(++lines_Mf_ptr) = cy;
		*(++lines_Mf_ptr) = length;
	}
}

void dect_opencv_lines(cv::Mat img, cv::Mat& lines_Mf_,float scale,int maxLineNum)
{
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
	//cv::Mat gray_mat;
	//cv::cvtColor(img, gray_mat, cv::COLOR_BGR2GRAY);
	std::vector<cv::Vec4f> lines_std;
	ls->detect(img, lines_std);
	float x1, y1, x2, y2, cx, cy, length;
	cv::Mat lines_Mf = cv::Mat(lines_std.size(), 7, CV_32FC1);
	float* lines_Mf_ptr = (float*)lines_Mf.data - 1;

	
	for (int i = 0; i < lines_std.size(); i++)
	{
		x1 = lines_std[i][0] / scale;
		y1 = lines_std[i][1] / scale;
		x2 = lines_std[i][2] / scale;
		y2 = lines_std[i][3] / scale;

		cx = (x1 + x2) / 2.0;
		cy = (y1 + y2) / 2.0;
		length = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
		length = std::sqrt(length);

		*(++lines_Mf_ptr) = x1;
		*(++lines_Mf_ptr) = y1;
		*(++lines_Mf_ptr) = x2;
		*(++lines_Mf_ptr) = y2;
		*(++lines_Mf_ptr) = cx;
		*(++lines_Mf_ptr) = cy;
		*(++lines_Mf_ptr) = length;

		
	}
	

	if (lines_Mf.rows < maxLineNum)
	{
		lines_Mf.copyTo(lines_Mf_);
		return;
	}

	cv::Mat order;
	cv::sortIdx(lines_Mf.colRange(6, 7), order, cv::SORT_DESCENDING + cv::SORT_EVERY_COLUMN);

	for (int i = 0; i < maxLineNum; i++)
	{
		lines_Mf_.push_back(lines_Mf.row(order.at<int>(i, 0)));
	}

}

//EDLine
void lineDetect_EDLine(cv::Mat in_img, cv::Mat& lines_Mf_)
{
	EDLineDetector edlined;

	edlined.EDline(in_img, true);
	edlined.lineEndpoints_;

	float x1, y1, x2, y2, cx, cy, length;
	cv::Mat lines_Mf = cv::Mat(edlined.lineEndpoints_.size(), 7, CV_32FC1);
	float* lines_Mf_ptr = (float*)lines_Mf.data - 1;
	for (int i = 0; i < edlined.lineEndpoints_.size(); i++)
	{
		x1 = edlined.lineEndpoints_[i][0] ;
		y1 = edlined.lineEndpoints_[i][1];
		x2 = edlined.lineEndpoints_[i][2];
		y2 = edlined.lineEndpoints_[i][3];

		cx = (x1 + x2) / 2.0;
		cy = (y1 + y2) / 2.0;
		length = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
		length = std::sqrt(length);

		*(++lines_Mf_ptr) = x1;
		*(++lines_Mf_ptr) = y1;
		*(++lines_Mf_ptr) = x2;
		*(++lines_Mf_ptr) = y2;
		*(++lines_Mf_ptr) = cx;
		*(++lines_Mf_ptr) = cy;
		*(++lines_Mf_ptr) = length;
	}
	lines_Mf.copyTo(lines_Mf_);
}


int processImage(
	SfMManager* sfm, int img_ind,
	float costhre, float dist, int inter_support_num, int maxwidth, int uselsd, int maxlineNum,
	std::string lineFolder, std::string lineExt)
{

	//1 read images
	std::string image_name = sfm->iImageNames(img_ind);
	std::string input_folder = sfm->inputFolder();
	cv::Mat img = cv::imread(input_folder + "/" + image_name, 0);

	// scale images
	// check image size
	int max_dim = std::max(img.rows, img.cols);
	unsigned int new_width = img.cols;
	unsigned int new_height = img.rows;

	sfm->addImSize(img.rows, img.cols, img_ind);
	

	cv::Mat lines_Mf;
	
	cv::Mat imgResized;
	float scale;

	if (maxwidth > 0 && max_dim > maxwidth)
	{
		// rescale
		scale = float(maxwidth) / float(max_dim);
		cv::resize(img, imgResized, cv::Size(), scale, scale);
	}
	else
	{
		imgResized = img;
		scale = 1; // for LSD

	}

	if (uselsd==1)
	{
		dect_opencv_lines(imgResized, lines_Mf, scale, maxlineNum);
		
	}
	else if (uselsd == 2)
	{
		dect_ag3_lines(imgResized, lines_Mf, scale);
		cv::Mat img_float;
		img.convertTo(img_float, CV_32FC1);
		allignLineSegments(&lines_Mf, img_float);

	}
	else if (uselsd == 3)
	{
		lineDetect_EDLine(imgResized, lines_Mf);
		printf("\n image line size: %d \n", lines_Mf.rows);
		cv::Mat img_float;
		img.convertTo(img_float, CV_32FC1);
		allignLineSegments(&lines_Mf, img_float);
	}
	else if(uselsd == 4)
	{
		string linestr = input_folder + "/" + image_name+".deeplsd";
		readLsdLines(lines_Mf,linestr, maxlineNum);
		cv::Mat img_float;
		img.convertTo(img_float, CV_32FC1);
		allignLineSegments(&lines_Mf, img_float);
	}
	else if (uselsd = -1) {

		std::string tempName = image_name;
		tempName.replace(tempName.find_last_of("."), tempName.length(), "");
		string linestr = lineFolder + "/" + tempName + lineExt;
		readLsdLines(lines_Mf, linestr, maxlineNum);

		if (lines_Mf.rows < 5) return 0;;

		cv::Mat img_float;
		cv::Mat img = cv::imread(input_folder + "/" + image_name, 0);
		img.convertTo(img_float, CV_32FC1);
		allignLineSegments(&lines_Mf, img_float);

	}
	
	if (lines_Mf.rows <5)	
		return 0;
	
	//3 detect intersection
	cv::Mat inter_lines_Mf;
	cv::Mat parra_lines_Mf;

	parallel_2lines((float*)lines_Mf.data, lines_Mf.rows, 0.9962, 5, 20, parra_lines_Mf);

	parra_lines_Mf.copyTo(*(sfm->iParraLines(img_ind)));

	callCrossPt(&inter_lines_Mf, (float*)lines_Mf.data, lines_Mf.rows, costhre, dist);

	if (inter_lines_Mf.rows <5)
	{
		lines_Mf.copyTo(*(sfm->iImageLines(img_ind)));
		return 0;
	}
	
	//4 load points and query knn points for intersections
	cv::Mat pt2 = *(sfm->iImagePointsPtr(img_ind));
	if (pt2.rows <= SUPPORT_POINT_NUM)
		return 0;


	//printf("points: %d\n", pt2.rows);
	cv::Mat pt2_Mf = pt2.colRange(0, 2);
	cv::Mat  pt2_idpt3__Mf = pt2.colRange(2, 3);

	// Allocate memory for computed k-NN neighbors
	float imr_2 = img.rows / 2.0;
	float imc_2 = img.cols / 2.0;

	if (sfm->sfmType=="vsfm")
	{
		for (int i = 0; i < pt2_Mf.rows; i++)
		{
			pt2_Mf.at<float>(i, 0) = pt2_Mf.at<float>(i, 0) + imc_2;
			pt2_Mf.at<float>(i, 1) = pt2_Mf.at<float>(i, 1) + imr_2;
		}
	}


	cv::Mat inter_knn_Mi, line_knn_Mi_1, line_knn_Mi_2, linemid_knn;

	if (!points_knn(inter_lines_Mf.colRange(2, 4).clone(),
		pt2_Mf, pt2_idpt3__Mf, SUPPORT_POINT_NUM, &inter_knn_Mi))
		return 0;

	if (!points_knn(lines_Mf.colRange(0, 2).clone(),
		pt2_Mf, pt2_idpt3__Mf, SUPPORT_POINT_NUM, &line_knn_Mi_1))
		return 0;

	if (!points_knn(lines_Mf.colRange(2, 4).clone(),
		pt2_Mf, pt2_idpt3__Mf, SUPPORT_POINT_NUM, &line_knn_Mi_2))
		return 0;

	
	sfm->setImageLineSize(lines_Mf.rows, img_ind);
	//5 compute the point depth for line and inter
	//5.1 calculate cameras
	//sfm->addCameraBySize(img.rows, img.cols, img_ind);

	cv::Mat inter_max_min = cv::Mat(inter_lines_Mf.rows, 2, CV_32FC1);
	maxMinPt(inter_lines_Mf.rows, inter_knn_Mi.cols, sfm->points3D_ptr(),
		(int*)inter_knn_Mi.data, sfm->iCameraMatPtr(img_ind), (float*)inter_max_min.data);

	cv::Mat line_max_min_1 = cv::Mat(lines_Mf.rows, 2, CV_32FC1);
	cv::Mat line_max_min_2 = cv::Mat(lines_Mf.rows, 2, CV_32FC1);

	maxMinPt(lines_Mf.rows, line_knn_Mi_1.cols, sfm->points3D_ptr(),
		(int*)line_knn_Mi_1.data, sfm->iCameraMatPtr(img_ind), (float*)line_max_min_1.data);

	maxMinPt(lines_Mf.rows, line_knn_Mi_2.cols, sfm->points3D_ptr(),
		(int*)line_knn_Mi_2.data, sfm->iCameraMatPtr(img_ind), (float*)line_max_min_2.data);

	cv::Mat line_max_min;
	hconcat(line_max_min_1, line_max_min_2, line_max_min);
	cv::Mat lines_ranges;
	hconcat(lines_Mf, line_max_min, *(sfm->iImageLines(img_ind)));
	cv::Mat inter_ranges;
	hconcat(inter_lines_Mf, inter_max_min, *(sfm->iJunctionLines(img_ind)));

	std::cout << "\n" << sfm->camsNumber() - img_ind << " images to process\n";
	
	return lines_Mf.rows;
}
