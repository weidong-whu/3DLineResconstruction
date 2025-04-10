#pragma once
#include <iostream>
#include <vector>
#include<string>
#include <opencv2/core.hpp>
#include"SfMManager.h"


void parallel_2lines(float* lines, int size, float costhre, float distmin, float distmax, cv::Mat& ppl_);

void callCrossPt(cv::Mat* lp_Mf, float* lines, int size, float costhre, float dist);

int processImage(SfMManager* sfm, int i,
	float costhre, float dist, int inter_support_num, int maxwidth, std::string lineType,int maxlineNum);