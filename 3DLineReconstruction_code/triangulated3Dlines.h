#ifndef TRIANGULATED3DLINE_HEADER

#include <math.h>
#include <iostream>

bool tringulate3Dline(
	float* cam2, float* M2, float* C2, float* line2,
	float* M1, float* C1, float* line1,
	float* pt3d1, float* pt3d2, float* maxcos, float* mindis);

#endif // !TRIANGULATED3DLINE_HEADER
