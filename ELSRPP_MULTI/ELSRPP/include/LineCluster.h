//
//  LineCluster.h
//  line cluster
//
//  Created by Alexxxxx on 2024/7/14.
//
#ifndef LINECLUSTER_PROGRAM
#define LINECLUSTER_PROGRAM

#include <iostream>
#include <filesystem>
#include <fstream>
#include <istream>
#include <random>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/normal.hpp>
#include <nlopt.hpp>


#include "nfa.h"
#include "LineSweep.h"

void readSFM(std::string outfolder, SFM_INFO& sfmInfo, IMG_INFO& imgInfo);
void normalBuild(std::string outfolder, SFM_INFO& sfmInfo, IMG_INFO& imgInfo, ARR_INFO& arrInfo);
void normalBuild(std::vector<cv::Mat> multiPoints, SFM_INFO& sfmInfo, IMG_INFO& imgInfo, ARR_INFO& arrInfo);
void callAdaptiveLineCluster(SFM_INFO& sfmInfo, IMG_INFO& imgInfo, ARR_INFO& arrInfo, SPACE_REC& spaceRec);
cv::Mat colinearRefine(IMG_INFO imgInfo, SPACE_REC spaceRec, PARAMS param);
cv::Mat coplanarRef(IMG_INFO imgInfo, SPACE_REC spaceRec, PARAMS param);
cv::Mat lineGrow1(IMG_INFO imgInfo, SPACE_REC spaceRec, cv::Mat control3D, cv::Mat reviseID, PARAMS param);
cv::Mat lineGrow2(IMG_INFO imgInfo, SPACE_REC spaceRec, cv::Mat revised3D, cv::Mat reviseID, PARAMS param);
cv::Mat extractMaxLine(cv::Mat line1, cv::Mat line2);
cv::Mat multiReconstruction(cv::Mat lines3D, IMG_INFO imgInfo, SPACE_REC spaceRec, PARAMS param);
void outObj(std::string filePath, cv::Mat lines);
void divideSpaceLine(cv::Mat& r_control3D, cv::Mat& r_reviseID, cv::Mat colinearRef, cv::Mat planeRef);

void lineCluster(SfMManager* sfm, MergeProcess* mergeProc, std::string inputFolder);

#endif // !LINECLUSTER_PROGRAM