#pragma once
#include <time.h>
#include <queue>
#include <vector>
#include"BaseLine.h"
#include "GeneralFuncs.h"
#include"RegManager.h"
#include <cmath>
struct lineag
{
	float x1, y1, x2, y2; /* first and second Point3i of the line segment */

};
int ag3line(cv::Mat& im, vector<lineag>& lines, bool control);
