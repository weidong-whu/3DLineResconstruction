#include"LineSweep.h"
#include<vector>
#include <numeric>
#include"IO.h"
#include "ThreadPool.h"
//#include"ReconstructLines.h"
#include"BasicMath.h"
#include "Parameters.h"
#include <opencv2/core/core_c.h>


#define LINE3D_VEC_COS 0.984807753012208 
#define MAX_PROJ_DIS 2.5 //px

bool isChecked(std::vector<int>& checkArr, int lineID) {


	for (int i = 0; i < checkArr.size(); i++) {
		if (checkArr[i] == lineID)
			return true;
	}


	checkArr.push_back(lineID);


	return false;

}



void map2Camera(float* CM, float* x, float* res_3_1) {
	res_3_1[0] = CM[3] + CM[0] * x[0] + CM[1] * x[1] + CM[2] * x[2];
	res_3_1[1] = CM[7] + CM[4] * x[0] + CM[5] * x[1] + CM[6] * x[2];
	res_3_1[2] = CM[11] + CM[8] * x[0] + CM[9] * x[1] + CM[10] * x[2];
}




void bresenhamMap(int x1, int y1, int const x2, int const y2, int imr, int imc, ushort* map, ushort mark) {
	//prepare for bresenham
	int delta_x(x2 - x1);
	signed char const ix((delta_x > 0) - (delta_x < 0));
	delta_x = std::abs(delta_x) << 1;

	int delta_y(y2 - y1);
	signed char const iy((delta_y > 0) - (delta_y < 0));
	delta_y = std::abs(delta_y) << 1;

	int error;
	if (delta_x >= delta_y)
		error = (delta_y - (delta_x >> 1));
	else
		error = (delta_x - (delta_y >> 1));

	while (BresenhamItera(x1, y1, x2, y2, ix, iy, delta_x, delta_y, error))
		if (x1 >= 0 && y1 >= 0 && x1 < imc && y1 < imr)
			map[imc * y1 + x1] = mark;


}



float checkLine2D(float* line2d, float* xp, float* yp) {
	float l[3];
	cross_v3(xp, yp, l);

	float d1, d2;

	d1 = point_2_line_dis(line2d[0], line2d[1], l);
	d2 = point_2_line_dis(line2d[2], line2d[3], l);


	if (d1 > d2)
		return d1;
	else
		return d2;



}

cv::Mat MergeProcess::createLineMap(int imageid) {

	cv::Mat* imageline = sfmM->iImageLines(imageid);

	int imr = 0, imc = 0;

	sfmM->iImSize(imageid, imr, imc);

	cv::Mat map = cv::Mat::zeros(imr, imc, CV_16UC1);
	ushort* mapptr = (ushort*)map.data;

	float* line = (float*)(imageline->data);

	for (int lineID = 0; lineID < imageline->rows; lineID++) {
		//printf("lineID:%d\n", lineID);

		int x1 = round(line[0]);
		int y1 = round(line[1]);
		int x2 = round(line[2]);
		int y2 = round(line[3]);
		//printf("%hd %f %f %f %f\n", lineID,x1, y1, y1,y2);
		bresenhamMap(x1, y1, x2, y2, imr, imc, mapptr, lineID + 1);

		line = line + imageline->cols;
	}

	return map;
}

void MergeProcess::sweepCheckLine(const ushort* limap, int imageid, int curmatch, float max_error, float mean, float std) {
	cv::Mat* lines3dM = matchM->line3DArrPtr(curmatch);
	cv::Mat* lines2dM = sfmM->iImageLines(imageid);

	float* lines3 = (float*)lines3dM->data;
	float* M = sfmM->iCamera33TransPtr(imageid);
	float* C = sfmM->iCameraCenterPtr(imageid);
	float* cam = sfmM->iCameraMatPtr(imageid);

	float xp[3], yp[3];

	int imc = 0, imr = 0;

	sfmM->iImSize(imageid, imr, imc);

	std::vector<int>checkarr;

	for (int matchID = 0; matchID < lines3dM->rows; matchID++) {


		ushort checkIDArr[50];
		int checkCounter = 0;


		int mark, bestLineID, bestLineID2;

		float bestDis = 999;
		float bestDis2 = 999;
		bestLineID = -1;
		bestLineID2 = -1;



		float vec31[3], vec32[3];

		vec31[0] = lines3[0] - lines3[3];
		vec31[1] = lines3[1] - lines3[4];
		vec31[2] = lines3[2] - lines3[5];



		//printf("lines read\n");
		map2Camera(cam, lines3, xp);
		map2Camera(cam, lines3 + 3, yp);
		//printf("%f %f %f %f %f %f \n", line1[0], line1[1], line1[2], line1[3], line1[4], line1[5]);

		lines3 = lines3 + lines3dM->cols;

		xp[0] = xp[0] / xp[2];
		xp[1] = xp[1] / xp[2];
		yp[0] = yp[0] / yp[2];
		yp[1] = yp[1] / yp[2];

		// very important
		xp[2] = 1;
		yp[2] = 1;

		float vec[2], vec1[2], vec_mark2[2];
		vec[0] = yp[0] - xp[0];
		vec[1] = yp[1] - xp[1];
		vec1[0] = vec[0];
		vec1[1] = vec[1];

		int x1 = round(xp[0]);
		int y1 = round(xp[1]);
		int x2 = round(yp[0]);
		int y2 = round(yp[1]);

		if (vec[0] > vec[1]) {
			vec[0] = 0;
			vec[1] = 1;
		}
		else {
			vec[0] = 1;
			vec[1] = 0;
		}

		int curx, cury;
		if (abs(x1 - x2) > imc || abs(y1 - y2) > imr)
			continue;

		//prepare for bresenham
		int delta_x(x2 - x1);
		signed char const ix((delta_x > 0) - (delta_x < 0));
		delta_x = std::abs(delta_x) << 1;

		int delta_y(y2 - y1);
		signed char const iy((delta_y > 0) - (delta_y < 0));
		delta_y = std::abs(delta_y) << 1;

		int error;
		if (delta_x >= delta_y)
			error = (delta_y - (delta_x >> 1));
		else
			error = (delta_x - (delta_y >> 1));

		checkarr.clear();

		while (BresenhamItera(x1, y1, x2, y2, ix, iy, delta_x, delta_y, error)) {
			for (int k = -MAX_PROJ_DIS; k <= MAX_PROJ_DIS; k++) {
				curx = x1 + k * vec[0];
				cury = y1 + k * vec[1];

				//printf("curx:%d cury:%d \n", curx, cury);
				if (!(curx >= 0 && cury >= 0 && curx < imc && cury < imr))
					continue;
				// is checked validation
				//printf("check mark in \n");
				mark = (int)(limap[imc * cury + curx]) - 1;
				if (isChecked(checkarr, mark))
					continue;

				float* line2d = (float*)lines2dM->data + mark * lines2dM->cols;

				vec_mark2[0] = line2d[2] - line2d[0];
				vec_mark2[1] = line2d[3] - line2d[1];

				if (vec_mark2[0] * vec1[0] + vec_mark2[1] * vec1[1] < 0)
					continue;

				// intersect check 
				if (!twoLines_intersec(xp, yp, line2d, line2d + 2, 0.5))
					continue;
				// line 2 line distance  
				float l2l_dis = checkLine2D(line2d, xp, yp);

				// line 3d check            
				if (l2l_dis > MAX_PROJ_DIS)
					continue;




				if (l2l_dis < bestDis) {
					bestDis = l2l_dis;
					bestLineID = mark;
				}
			}
		}



		//check if exist

		if (bestLineID == -1) {
			continue;
		}

		camsID[curmatch][matchID].push_back(imageid);
		lineID[curmatch][matchID].push_back(bestLineID);
		errors[curmatch][matchID].push_back(abs((bestDis - mean)) / std);
	}
}



void MergeProcess::beginSweep() {

	int conectionNum = 50;
	std::vector<int>matchID_Vec;
	for (int imageID = 0; imageID < sfmM->camsNumber(); imageID++) {
		int lineSize = sfmM->iImageLineSize(imageID);
		if (lineSize <= minimumCellCout)
			continue;

		int matchIM1, matchIM2, connectNum1, connectNum2;
		std::vector<int> matchID_Vec;
		matchID_Vec.clear();
		for (int matchID = 0; matchID < matchM->matchSize(); matchID++) {

			matchM->iPairIndex(matchID, matchIM1, matchIM2);

			if (matchIM1 == imageID || matchIM2 == imageID)
				continue;

			matchM->image2imageScore(imageID, matchIM1, connectNum1);
			matchM->image2imageScore(imageID, matchIM2, connectNum2);
			if (connectNum1 < conectionNum || connectNum2 < conectionNum)
				continue;

			matchID_Vec.push_back(matchID);

		}

		printf("sweeping %d pairs ", matchID_Vec.size());

		sweep4Image(imageID, matchID_Vec);

		printf("%d images left for sweeping\n", sfmM->camsNumber() - imageID);

	}
}

void MergeProcess::sweep4Image(int imageID, std::vector<int> matchids) {

	cv::Mat limap = createLineMap(imageID);
	ushort* limapptr = (ushort*)limap.data;

	float max_error = 0, mean = 0, std = 0;
	sfmM->getTrainCell(imageID, mean, std, max_error);

#pragma omp parallel for
	for (int i = 0; i < matchids.size(); i++) {
		int matchID = matchids[i];
		cv::Mat* curLines3D = (matchM->line3DArrPtr(matchID));
		int matchSize = curLines3D->rows;
		if (matchSize <= minimumCellCout)
			continue;

		sweepCheckLine(limapptr, imageID, matchID, max_error, mean, std);

	}

}



MergeProcess::MergeProcess(SfMManager* sfmM, MatchManager* matchM) {


	this->sfmM = sfmM;
	this->matchM = matchM;

	camsID.resize(matchM->matchSize());
	lineID.resize(matchM->matchSize());
	errors.resize(matchM->matchSize());
	//pairID.resize(matchM->matchSize());

	for (int i = 0; i < matchM->matchSize(); i++) {
		int n = matchM->line3DArrPtr(i)->rows;

		if (n <= minimumCellCout)
			continue;

		camsID[i].resize(n);
		lineID[i].resize(n);
		errors[i].resize(n);
	}

	int im1 = 0, im2 = 0;
	int lid1 = 0, lid2 = 0;
	for (int i = 0; i < matchM->matchSize(); i++) {
		cv::Mat* matchMi = matchM->matcheArrPtr(i);
		ushort* matchMiptr = (ushort*)matchMi->data;
		matchM->iPairIndex(i, im1, im2);

		int n = matchM->line3DArrPtr(i)->rows;

		if (n <= minimumCellCout)
			continue;

		for (int j = 0; j < matchMi->rows; j++) {
			lid1 = matchMiptr[0];
			lid2 = matchMiptr[2];

			camsID[i][j].push_back(im1);
			camsID[i][j].push_back(im2);

			lineID[i][j].push_back(lid1);
			lineID[i][j].push_back(lid2);

			matchMiptr = matchMiptr + matchMi->cols;
		}
	}
}




