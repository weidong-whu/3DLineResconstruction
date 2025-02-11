#pragma once
#include<vector>
#include<math.h>

float norm_no_sqrt(float x, float y);

float ang_of_vec(float vx1, float vy1, float vx2, float vy2);

float norm_v3(float* v3);

float normM(float x, float y);

float det(float a, float b, float c, float d);

float dot(float a, float b, float c, float d);

float cos_vec(float* veci, float* vecj);

float cos_vec3(float* veci, float* vecj);

float dot_v2(float x1, float y1, float x2, float y2);

float point_2_line_dis_3D(float* x_0, float* x_1, float* x_2);

float dot_v3(float* v1, float* v2);

float point_2_line_dis(float x, float y, float* l);

float point_2_line_dis(float* pt, float* linef);

bool twoLines_intersec(float* pt1, float* pt2, float* tl1, float* tl2, float intersecratio);

void mult_3_3_1(float* o, float* e, float* res_3_1);

void mult_3_4_4(float* CM, float* x, float* res_3_1);

float norm_v2(float* v2);

void cross_v3(float* v1, float* v2, float* v3);

void cross_v2(float* line, float* linef);

void mult_3_3_3(float* o, float* e, float* res_3_1);

void M_divide_b(float* M, float* b, float* v);

void norm_by_v3(float* v);

void norm_by_v4(float* v4);

void Bresenham(int x1,
	int y1,
	int const x2,
	int const y2,
	std::vector<int>& xx, std::vector<int>& yy);

void Bresenham(int x1,
	int y1,
	int const x2,
	int const y2,
	int* xx, int* yy, int& xy_size);

float max_2(float a, float b);

float min_2(float a, float b);

float max_4(float a, float b, float c, float d);

bool ID_in_array(int* ids, int t_size, int num, int allocate_num);



bool tringulate3Dline(
	float* cam2, float* line2,
	float* M1, float* C1, float* line1,
	float* pt3d1, float* pt3d2);

void rayInterPlane(float* rayVector, float* rayPoint, float* planeNormal, float* planePoint, float* interPoint);

void solverAxb(const float M[9], const float itp[2], float pv[3]);
void equation_plane(float x1, float y1,
	float z1, float x2,
	float y2, float z2,
	float x3, float y3, float z3, float* vv);

float epipolarAngleCos(float* line1, float* line2, float* cam1, float* cam2);

bool BresenhamItera(int& x1,
	int& y1,
	int const x2,
	int const y2,
	signed char const& ix,
	signed char const& iy,
	int& delta_x,
	int& delta_y,
	int& error);

float linePlaneSin(float* pt1, float* pt2, float* plane);


float projectPt2Plane_dis(float* pt, float* plane);
float pt3dLength(float* pt1, float* pt2);
float point_plane_dis3d(float* pt, float* plane);

float closestPointAlongLine(float* lineFrom, float* line_vec, float* point, float* nearestPoint);

float projectPt2Plane(float* pt, float* plane);

void line2_spaceplane(float* l, float* cam, float* plane);

float vectorPlaneSin_cpu(float* vv, float* plane);