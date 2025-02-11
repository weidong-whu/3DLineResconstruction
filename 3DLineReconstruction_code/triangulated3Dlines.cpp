#include "triangulated3Dlines.h"
#include "BasicMath.h"

#ifndef LINE_plane_sin_min
#define LINE_plane_sin_min 0.0087
#endif
bool tringulate3Dline(
	float* cam2, float* M2, float* C2, float* line2,
	float* M1, float* C1, float* line1,
	float* pt3d1, float* pt3d2, float* maxcos, float* mindis)
{

	float* lpt1 = line1;
	float* lpt2 = line1 + 2;

	float* rpt1 = line2;
	float* rpt2 = line2 + 2;

	//plane
	float plane[4];
	float lf2[3];

	cross_v2(line2, lf2);
	line2_spaceplane(lf2, cam2, plane);

	float ray1[3], ray2[3];

	//ray1
	solverAxb(M1, lpt1, ray1);

	if (1)
	{
		float sina = vectorPlaneSin_cpu(ray1, plane);
		// printf("sina %f\n",sina);
		if (sina < LINE_plane_sin_min)
			return false;
	}

	//ray and plane intersection
	float planePoint[3] = { 0,0,-plane[3] / plane[2] };
	rayInterPlane(ray1, C1, plane, planePoint, pt3d1);

	//ray2
	solverAxb(M1, lpt2, ray2);

	if (1)
	{
		float sina = vectorPlaneSin_cpu(ray2, plane);
		// printf("sina %f\n",sina);
		if (sina < LINE_plane_sin_min)
			return false;
	}

	//ray and plane intersection
	rayInterPlane(ray2, C1, plane, planePoint, pt3d2);

	float lineray[3];
	lineray[0] = pt3d2[0] - pt3d1[0];
	lineray[1] = pt3d2[1] - pt3d1[1];
	lineray[2] = pt3d2[2] - pt3d1[2];
	float cos1 = cos_vec3(lineray, ray1);
	float cos2 = cos_vec3(lineray, ray2);

	if (cos1 < 0)
		cos1 = -cos1;
	if (cos2 < 0)
		cos2 = -cos2;


	// point 2 plane dis
	float vv[4];
	equation_plane(C1[0], C1[1], C1[2], C2[0], C2[1], C2[2], pt3d1[0], pt3d1[1], pt3d1[2], vv);
	float p2p1 = point_plane_dis3d(pt3d2, vv);
	equation_plane(C1[0], C1[1], C1[2], C2[0], C2[1], C2[2], pt3d2[0], pt3d2[1], pt3d2[2], vv);
	float p2p2 = point_plane_dis3d(pt3d1, vv);

	*maxcos = max_2(cos1, cos2);
	*mindis = min_2(p2p1, p2p2);


	return true;

}