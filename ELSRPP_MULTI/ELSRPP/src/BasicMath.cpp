#pragma once
#include "BasicMath.h"
#include "Parameters.h"
#include <iostream>
#include <math.h>
#include <vector>

float ang_of_vec(float vx1, float vy1, float vx2, float vy2)
{
    float dot = vx1 * vx2 + vy1 * vy2; // dot product
    float det = vx1 * vy2 - vy1 * vx2; // determinant
    return atan2(det, dot);
}

float norm_v3(float *v3)
{
    return sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2]);
}

float normM(float x, float y)
{
    return sqrt(x * x + y * y);
}

float norm_no_sqrt(float x, float y)
{
    return (x * x + y * y);
}

float det(float a, float b, float c, float d)
{
    return a * d - b * c;
}

float dot(float a, float b, float c, float d)
{
    return a * c + b * d;
}

float cos_vec3(float *veci, float *vecj)
{
    return (veci[0] * vecj[0] + veci[1] * vecj[1] + veci[2] * vecj[2]) / norm_v3(veci) / norm_v3(vecj);
}

float cos_vec(float *veci, float *vecj)
{
    return (veci[0] * vecj[0] + veci[1] * vecj[1]) /
           (sqrt(veci[0] * veci[0] + veci[1] * veci[1]) * sqrt(vecj[0] * vecj[0] + vecj[1] * vecj[1]));
}

float dot_v2(float x1, float y1, float x2, float y2)
{

    return x1 * x2 + y1 * y2;
}

float point_2_line_dis(float x, float y, float *l)
{
    return abs(x * l[0] + y * l[1] + l[2]) / sqrt(l[0] * l[0] + l[1] * l[1]);
}

// 3D point to line distance
// d   =   (|(x_2-x_1)x(x_1-x_0)|)/(|x_2-x_1|)
// https://stackoverflow.com/questions/19878441/point-line-distance-calculation
float point_2_line_dis_3D(float *x_0, float *x_1, float *x_2)
{
    float x2_x1[3], x1_x0[3], cv3[3];

    x2_x1[0] = x_2[0] - x_1[0];
    x2_x1[1] = x_2[1] - x_1[1];
    x2_x1[2] = x_2[2] - x_1[2];

    x1_x0[0] = x_1[0] - x_0[0];
    x1_x0[1] = x_1[1] - x_0[1];
    x1_x0[2] = x_1[2] - x_0[2];

    cross_v3(x2_x1, x1_x0, cv3);

    return norm_v3(cv3) / norm_v3(x2_x1);
}

// 2D point to line distance
float point_2_line_dis(float *pt, float *linef)
{
    return abs(pt[0] * linef[0] + pt[1] * linef[1] + linef[2]) / sqrt(linef[0] * linef[0] + linef[1] * linef[1]);
}

bool twoLines_intersec(float *pt1, float *pt2, float *tl1, float *tl2, float intersecratio)
{
    bool dottl_p1, dottl_p2, dotp_tl1, dotp_tl2;

    dottl_p1 = dot_v2(tl1[0] - pt1[0], tl1[1] - pt1[1], tl2[0] - pt1[0], tl2[1] - pt1[1]) <= 0;
    dottl_p2 = dot_v2(tl1[0] - pt2[0], tl1[1] - pt2[1], tl2[0] - pt2[0], tl2[1] - pt2[1]) <= 0;
    dotp_tl1 = dot_v2(pt1[0] - tl1[0], pt1[1] - tl1[1], pt2[0] - tl1[0], pt2[1] - tl1[1]) <= 0;
    dotp_tl2 = dot_v2(pt1[0] - tl2[0], pt1[1] - tl2[1], pt2[0] - tl2[0], pt2[1] - tl2[1]) <= 0;

    if (!(dottl_p1 || dottl_p2 || dotp_tl1 || dotp_tl2))
        return false;

    if ((dottl_p1 && dottl_p2) || (dotp_tl1 && dotp_tl2))
        return true;

    // pt1 is included
    if (dottl_p1)
    {
        float t_len;
        float l1_len = sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]));
        float l2_len = sqrt((tl1[0] - tl2[0]) * (tl1[0] - tl2[0]) + (tl1[1] - tl2[1]) * (tl1[1] - tl2[1]));
        t_len = sqrt((pt1[0] - tl2[0]) * (pt1[0] - tl2[0]) + (pt1[1] - tl2[1]) * (pt1[1] - tl2[1]));

        if (t_len / l1_len < intersecratio && t_len / l2_len < intersecratio)
            return 0;
    }

    if (dottl_p2)
    {
        float t_len;
        float l1_len = sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]));
        float l2_len = sqrt((tl1[0] - tl2[0]) * (tl1[0] - tl2[0]) + (tl1[1] - tl2[1]) * (tl1[1] - tl2[1]));

        t_len = sqrt((pt2[0] - tl1[0]) * (pt2[0] - tl1[0]) + (pt2[1] - tl1[1]) * (pt2[1] - tl1[1]));

        if (t_len / l1_len < intersecratio && t_len / l2_len < intersecratio)
            return 0;
    }

    return true;
}

void mult_3_3_1(float *o, float *e, float *res_3_1)
{
    res_3_1[0] = e[0] * o[0] + e[1] * o[1] + e[2] * o[2];
    res_3_1[1] = e[0] * o[3] + e[1] * o[4] + e[2] * o[5];
    res_3_1[2] = e[0] * o[6] + e[1] * o[7] + e[2] * o[8];
}

void mult_3_4_4(float *CM, float *x, float *res_3_1)
{
    res_3_1[0] = CM[3] + CM[0] * x[0] + CM[1] * x[1] + CM[2] * x[2];
    res_3_1[1] = CM[7] + CM[4] * x[0] + CM[5] * x[1] + CM[6] * x[2];
    res_3_1[2] = CM[11] + CM[8] * x[0] + CM[9] * x[1] + CM[10] * x[2];
}

float norm_v2(float *v2)
{
    return sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
}

void cross_v3(float *v1, float *v2, float *v3)
{
    v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
    v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
    v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void cross_v2(float *line, float *linef)
{
    linef[0] = line[1] - line[3];
    linef[1] = line[2] - line[0];
    linef[2] = line[0] * line[3] - line[1] * line[2];
}

bool ID_in_array(int *ids, int t_size, int num, int allocate_num)
{

    for (int i = 0; i < t_size && i <= allocate_num; i++)
    {
        if (num == ids[i])
            return true;
    }

    return false;
}

void mult_3_3_3(float *o, float *e, float *res_3_1)
{
    res_3_1[0] = e[0] * o[0] + e[1] * o[1] + e[2] * o[2];
    res_3_1[1] = e[0] * o[3] + e[1] * o[4] + e[2] * o[5];
    res_3_1[2] = e[0] * o[6] + e[1] * o[7] + e[2] * o[8];
}

void M_divide_b(float *M, float *b, float *v)
{
    v[0] = -(M[1] * M[5] * b[2] - M[2] * M[4] * b[2] - M[1] * M[8] * b[1] + M[2] * M[7] * b[1] + M[4] * M[8] * b[0] -
             M[5] * M[7] * b[0]) /
           (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] -
            M[2] * M[4] * M[6]);

    v[1] = (M[0] * M[5] * b[2] - M[2] * M[3] * b[2] - M[0] * M[8] * b[1] + M[2] * M[6] * b[1] + M[3] * M[8] * b[0] -
            M[5] * M[6] * b[0]) /
           (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] -
            M[2] * M[4] * M[6]);

    v[2] = -(M[0] * M[4] * b[2] - M[1] * M[3] * b[2] - M[0] * M[7] * b[1] + M[1] * M[6] * b[1] + M[3] * M[7] * b[0] -
             M[4] * M[6] * b[0]) /
           (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7] -
            M[2] * M[4] * M[6]);
}

float dot_v3(float *v1, float *v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void norm_by_v4(float *v4)
{
    v4[0] = v4[0] / v4[3];
    v4[1] = v4[1] / v4[3];
    v4[2] = v4[2] / v4[3];
    v4[3] = 1;
}

void norm_by_v3(float *v)
{
    v[0] = v[0] / v[2];
    v[1] = v[1] / v[2];
    v[2] = 1;
}

void Bresenham(int x1, int y1, int const x2, int const y2, std::vector<int> &xx, std::vector<int> &yy)
{
    xx.clear();
    yy.clear();
    int delta_x(x2 - x1);
    // if x1 == x2, then it does not matter what we set here
    signed char const ix((delta_x > 0) - (delta_x < 0));
    delta_x = std::abs(delta_x) << 1;

    int delta_y(y2 - y1);
    // if y1 == y2, then it does not matter what we set here
    signed char const iy((delta_y > 0) - (delta_y < 0));
    delta_y = std::abs(delta_y) << 1;
    xx.push_back(x1);
    yy.push_back(y1);
    // plot(x1, y1);

    if (delta_x >= delta_y)
    {
        // error may go below zero
        int error(delta_y - (delta_x >> 1));

        while (x1 != x2)
        {
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            error += delta_y;
            x1 += ix;

            // plot(x1, y1);
            xx.push_back(x1);
            yy.push_back(y1);
        }
    }
    else
    {
        // error may go below zero
        int error(delta_x - (delta_y >> 1));

        while (y1 != y2)
        {
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing

            error += delta_x;
            y1 += iy;

            xx.push_back(x1);
            yy.push_back(y1);
        }
    }
}

bool BresenhamItera(int &x1, int &y1, int const x2, int const y2, signed char const &ix, signed char const &iy,
                    int &delta_x, int &delta_y, int &error)
{

    if (delta_x >= delta_y)
    {

        while (x1 != x2)
        {
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            error += delta_y;
            x1 += ix;

            // plot(x1, y1);
            return 1;
        }
    }
    else
    {

        while (y1 != y2)
        {
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing

            error += delta_x;
            y1 += iy;

            return 1;
        }
    }

    return 0;
}

void Bresenham(int x1, int y1, int const x2, int const y2, int *xx, int *yy, int &xy_size)
{
    xy_size = 0;

    int delta_x(x2 - x1);
    // if x1 == x2, then it does not matter what we set here
    signed char const ix((delta_x > 0) - (delta_x < 0));
    delta_x = std::abs(delta_x) << 1;

    int delta_y(y2 - y1);
    // if y1 == y2, then it does not matter what we set here
    signed char const iy((delta_y > 0) - (delta_y < 0));
    delta_y = std::abs(delta_y) << 1;
    xx[xy_size] = x1;
    yy[xy_size] = y1;
    xy_size++;

    if (delta_x >= delta_y)
    {
        // error may go below zero
        int error(delta_y - (delta_x >> 1));

        while (x1 != x2)
        {
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            error += delta_y;
            x1 += ix;

            xx[xy_size] = x1;
            yy[xy_size] = y1;
            xy_size++;
        }
    }
    else
    {
        // error may go below zero
        int error(delta_x - (delta_y >> 1));

        while (y1 != y2)
        {
            // reduce error, while taking into account the corner case of error == 0
            if ((error > 0) || (!error && (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing
            error += delta_x;
            y1 += iy;

            xx[xy_size] = x1;
            yy[xy_size] = y1;
            xy_size++;
        }
    }
}

float max_2(float a, float b)
{
    if (a < b)
        return b;
    else
        return a;
}

float min_2(float a, float b)
{
    if (a > b)
        return b;
    else
        return a;
}

float max_4(float a, float b, float c, float d)
{
    return max_2(max_2(a, b), max_2(c, d));
}

void line2_spaceplane(float *l, float *cam, float *plane)
{
    plane[0] = cam[0] * l[0] + cam[4] * l[1] + cam[8] * l[2];
    plane[1] = cam[1] * l[0] + cam[5] * l[1] + cam[9] * l[2];
    plane[2] = cam[2] * l[0] + cam[6] * l[1] + cam[10] * l[2];
    plane[3] = cam[3] * l[0] + cam[7] * l[1] + cam[11] * l[2];
}

// tested
void rayInterPlane(float *rayVector, float *rayPoint, float *planeNormal, float *planePoint, float *interPoint)
{
    float diff[3] = {rayPoint[0] - planePoint[0], rayPoint[1] - planePoint[1], rayPoint[2] - planePoint[2]};

    float prod1 = dot_v3(diff, planeNormal);
    float prod2 = dot_v3(rayVector, planeNormal);
    float prod3 = prod1 / prod2;

    interPoint[0] = rayPoint[0] - rayVector[0] * prod3;
    interPoint[1] = rayPoint[1] - rayVector[1] * prod3;
    interPoint[2] = rayPoint[2] - rayVector[2] * prod3;
}

float vectorPlaneSin_cpu(float *vv, float *plane)
{
    float v0, v1, v2;
    v0 = vv[0];
    v1 = vv[1];
    v2 = vv[2];
    float sin_ang = std::abs(plane[0] * v0 + plane[1] * v1 + plane[2] * v2) /
                    std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]) /
                    std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

    return sin_ang;
}

bool tringulate3Dline(float *cam2, float *line2, float *M1, float *C1, float *line1, float *pt3d1, float *pt3d2)
{

    float *lpt1 = line1;
    float *lpt2 = line1 + 2;

    float *rpt1 = line2;
    float *rpt2 = line2 + 2;

    // plane
    float plane[4];
    float lf2[3];

    cross_v2(line2, lf2);
    line2_spaceplane(lf2, cam2, plane);

    float ray[3];

    // ray1
    solverAxb(M1, lpt1, ray);

    if (0)
    {
        float sina = vectorPlaneSin_cpu(ray, plane);
        if (sina < LINE_plane_sin_min)
            return false;
    }

    // ray and plane intersection
    float planePoint[3] = {0, 0, -plane[3] / plane[2]};
    rayInterPlane(ray, C1, plane, planePoint, pt3d1);

    // ray2
    solverAxb(M1, lpt2, ray);

    if (0)
    {
        float sina = vectorPlaneSin_cpu(ray, plane);
        if (sina < LINE_plane_sin_min)
            return false;
    }

    // ray and plane intersection
    rayInterPlane(ray, C1, plane, planePoint, pt3d2);
    return true;
}

void solverAxb(const float M[9], const float itp[2], float pv[3])
{
    float A[9];
    float itpm[3] = {itp[0], itp[1], 1};
    float a21;
    float maxval;
    int r1;
    int r2;
    int r3;
    for (r1 = 0; r1 < 9; r1++)
    {
        A[r1] = M[r1];
    }
    r1 = 0;
    r2 = 1;
    r3 = 2;
    maxval = std::abs(M[0]);
    a21 = std::abs(M[1]);
    if (a21 > maxval)
    {
        maxval = a21;
        r1 = 1;
        r2 = 0;
    }
    if (std::abs(M[2]) > maxval)
    {
        r1 = 2;
        r2 = 1;
        r3 = 0;
    }
    A[r2] = M[r2] / M[r1];
    A[r3] /= A[r1];
    A[r2 + 3] -= A[r2] * A[r1 + 3];
    A[r3 + 3] -= A[r3] * A[r1 + 3];
    A[r2 + 6] -= A[r2] * A[r1 + 6];
    A[r3 + 6] -= A[r3] * A[r1 + 6];
    if (std::abs(A[r3 + 3]) > std::abs(A[r2 + 3]))
    {
        int rtemp;
        rtemp = r2;
        r2 = r3;
        r3 = rtemp;
    }
    A[r3 + 3] /= A[r2 + 3];
    A[r3 + 6] -= A[r3 + 3] * A[r2 + 6];
    pv[1] = itpm[r2] - itpm[r1] * A[r2];
    pv[2] = (itpm[r3] - itpm[r1] * A[r3]) - pv[1] * A[r3 + 3];
    pv[2] /= A[r3 + 6];
    pv[0] = itpm[r1] - pv[2] * A[r1 + 6];
    pv[1] -= pv[2] * A[r2 + 6];
    pv[1] /= A[r2 + 3];
    pv[0] -= pv[1] * A[r1 + 3];
    pv[0] /= A[r1];
}

// Function Definitions
void solverAxb3(const float A[9], const float b[3], float x[3])
{
    float b_A[9];
    float a21;
    float maxval;
    int r1;
    int r2;
    int r3;
    for (r1 = 0; r1 < 9; r1++)
    {
        b_A[r1] = A[r1];
    }
    r1 = 0;
    r2 = 1;
    r3 = 2;
    maxval = std::abs(A[0]);
    a21 = std::abs(A[1]);
    if (a21 > maxval)
    {
        maxval = a21;
        r1 = 1;
        r2 = 0;
    }
    if (std::abs(A[2]) > maxval)
    {
        r1 = 2;
        r2 = 1;
        r3 = 0;
    }
    b_A[r2] = A[r2] / A[r1];
    b_A[r3] /= b_A[r1];
    b_A[r2 + 3] -= b_A[r2] * b_A[r1 + 3];
    b_A[r3 + 3] -= b_A[r3] * b_A[r1 + 3];
    b_A[r2 + 6] -= b_A[r2] * b_A[r1 + 6];
    b_A[r3 + 6] -= b_A[r3] * b_A[r1 + 6];
    if (std::abs(b_A[r3 + 3]) > std::abs(b_A[r2 + 3]))
    {
        int rtemp;
        rtemp = r2;
        r2 = r3;
        r3 = rtemp;
    }
    b_A[r3 + 3] /= b_A[r2 + 3];
    b_A[r3 + 6] -= b_A[r3 + 3] * b_A[r2 + 6];
    x[1] = b[r2] - b[r1] * b_A[r2];
    x[2] = (b[r3] - b[r1] * b_A[r3]) - x[1] * b_A[r3 + 3];
    x[2] /= b_A[r3 + 6];
    x[0] = b[r1] - x[2] * b_A[r1 + 6];
    x[1] -= x[2] * b_A[r2 + 6];
    x[1] /= b_A[r2 + 3];
    x[0] -= x[1] * b_A[r1 + 3];
    x[0] /= b_A[r1];
}

float epipolarAngleCos(float *line1, float *line2, float *cam1, float *cam2)
{
    // plane1
    float plane1[4];
    float lf1[3];

    // plane2
    float plane2[4];
    float lf2[3];

    cross_v2(line1, lf1);
    line2_spaceplane(lf1, cam1, plane1);

    cross_v2(line2, lf2);
    line2_spaceplane(lf2, cam2, plane2);

    return abs(cos_vec3(plane1, plane2));
}

// Function to find equation of plane.
void equation_plane(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float *vv)
{
    float a1 = x2 - x1;
    float b1 = y2 - y1;
    float c1 = z2 - z1;
    float a2 = x3 - x1;
    float b2 = y3 - y1;
    float c2 = z3 - z1;

    vv[0] = b1 * c2 - b2 * c1;
    vv[1] = a2 * c1 - a1 * c2;
    vv[2] = a1 * b2 - b1 * a2;
    vv[3] = (-vv[0] * x1 - vv[1] * y1 - vv[2] * z1);
}

float linePlaneSin(float *pt1, float *pt2, float *plane)
{
    float v0, v1, v2;
    v0 = pt1[0] - pt2[0];
    v1 = pt1[1] - pt2[1];
    v2 = pt1[2] - pt2[2];
    float sin_ang = std::abs(plane[0] * v0 + plane[1] * v1 + plane[2] * v2) /
                    std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]) /
                    std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    if (isnan(sin_ang))
    {
        printf("%f %f %f\n", v0, v1, v2);
        getchar();
    }
    return sin_ang;
}

// revise0324 another metod for point 2 line projection
float projectPt2Plane_dis(float *pt, float *plane)
{
    float pt2[3];

    float xo = pt[0];
    float yo = pt[1];
    float zo = pt[2];

    float A = plane[0];
    float B = plane[1];
    float C = plane[2];
    float D = plane[3];

    float sum_2 = A * A + B * B + C * C;

    pt2[0] = ((B * B + C * C) * xo - A * (B * yo + C * zo + D)) / sum_2;
    pt2[1] = ((A * A + C * C) * yo - B * (A * xo + C * zo + D)) / sum_2;
    pt2[2] = ((A * A + B * B) * zo - C * (A * xo + B * yo + D)) / sum_2;

    return pt3dLength(pt, pt2);
}

float pt3dLength(float *pt1, float *pt2)
{
    float x_ = pt1[0] - pt2[0];
    float y_ = pt1[1] - pt2[1];
    float z_ = pt1[2] - pt2[2];

    return sqrt(x_ * x_ + y_ * y_ + z_ * z_);
}

float point_plane_dis3d(float *pt, float *plane)
{
    return std::abs(pt[0] * plane[0] + pt[1] * plane[1] + pt[2] * plane[2] + plane[3]) /
           std::sqrtf(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
}

float closestPointAlongLine(float *lineFrom, float *line_vec, float *point, float *nearestPoint)
{

    float vec_len = norm_v2(line_vec);
    line_vec[0] = line_vec[0] / vec_len;
    line_vec[1] = line_vec[1] / vec_len;

    float v[2];
    v[0] = point[0] - lineFrom[0];
    v[1] = point[1] - lineFrom[1];

    float w = dot_v2(line_vec[0], line_vec[1], v[0], v[1]);

    nearestPoint[0] = line_vec[0] * w + lineFrom[0];
    nearestPoint[1] = line_vec[1] * w + lineFrom[1];

    float diff[] = {nearestPoint[0] - point[0], nearestPoint[1] - point[1]};

    return sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
}

float projectPt2Plane(float *pt, float *plane)
{
    float t;

    float x = pt[0];
    float y = pt[1];
    float z = pt[2];

    float a = plane[0];
    float b = plane[1];
    float c = plane[2];
    float d = plane[3];

    float dist = (a * x + b * y + c * z + d) / sqrt(a * a + b * b + c * c);

    return std::abs(dist);
}