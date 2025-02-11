#pragma once
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include "IO.h"
#include "SfMManager.h"
#include "MatchManager.h"

int read_VisualSfM(
	SfMManager* sfm,
	MatchManager* match,
	int knn_image,
	int connectionNum,
	bool fromcolmap);

int read_VisualSfM_PS(
	SfMManager* sfm,
	MatchManager* match,
	int knn_image,
	int connectionNum,
	bool fromcolmap);


