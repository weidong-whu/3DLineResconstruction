#pragma once
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include "IO.h"
#include "SfMManager.h"
#include "MatchManager.h"


struct point_info {
	float xx;
	float yy;
	uint camid;
};

void point2lineerr(float* pt3, std::vector<point_info>& points, std::vector<cv::Mat>& Ms, std::vector<cv::Mat>& Cs, std::vector<cv::Mat*>& errs);

bool get_image_size_without_decode_image(const char* file_path, int* width, int* height, int* file_size);