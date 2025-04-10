#pragma once

#ifndef POINT_LINE_DIS
#define POINT_LINE_DIS 2
#endif 

#ifndef LINE_LINE_ANG
#define LINE_LINE_ANG 0.1745// pi/18 0.1745 pi/36 0.0872664625997 
#endif 

#ifndef INTERSECT_DIS
#define INTERSECT_DIS 15
#endif 

// for acceleration in finding the intersection
#ifndef INTERSECT_COS
#define INTERSECT_COS 0.8660
#endif 

#ifndef SUPPORT_POINT_NUM
#define SUPPORT_POINT_NUM 15
#endif 

#ifndef SUPPORT_HOMO_NUM
#define SUPPORT_HOMO_NUM 10
#endif 

#ifndef LINE_OVERLAP
#define LINE_OVERLAP 0.5
#endif 

#ifndef LINE_plane_sin_min
#define LINE_plane_sin_min 0.01750
#endif 



/*
  One can scale the image by change this threshold.
  it is set as a large number to avoid scale.
 */
#ifndef MAX_IMAGE_WIDTH
#define MAX_IMAGE_WIDTH 60000 //revise0505
#endif 


#ifndef ADAPTIVE_BINS
#define ADAPTIVE_BINS 25
#endif 

