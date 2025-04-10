#include "MatchManager.h"
#include "IO.h"
#include <iostream>
void MatchManager::image2imageScore(int im1, int im2, int &score)
{
    score = connectScore.at<ushort>(im1, im2);
}

void analysis_match(cv::Mat mscores, cv::Mat &imidx_Mf, std::vector<int> lineSize, int knn, int minconnect)
{

    int ind;

    cv::Mat used = cv::Mat::zeros(mscores.rows, mscores.cols, CV_16SC1);

    cv::Mat per_pair(1, 2, CV_32FC1);

    for (int i = 0; i < mscores.cols; i++)
    {
        cv::Mat sortID;
        cv::sortIdx(mscores.row(i), sortID, cv::SORT_DESCENDING);
        int cur = 0;
        for (int j = 0; j < knn; j++)
        {
            ind = sortID.at<int>(0, j);

            if (used.at<ushort>(ind, i) == 1 || used.at<ushort>(i, ind) == 1)
                continue;

            if (mscores.at<ushort>(i, ind) < minconnect)
                break;

            per_pair.at<float>(0, 0) = i;
            per_pair.at<float>(0, 1) = ind;
            imidx_Mf.push_back(per_pair);

            used.at<ushort>(ind, i) = 1;
            used.at<ushort>(i, ind) = 1;
        }
    }
}

void MatchManager::initializeM(std::vector<int> &lineSize, int knn, int minconnect)
{
    analysis_match(connectScore, pairIndex, lineSize, knn, minconnect);

    matchNums = pairIndex.rows;
    line3DArr.resize(matchNums);
    line3DArr_revise.resize(matchNums);
    matcheArr.resize(matchNums);
    checkPairs.resize(matchNums);

    ismatching.resize(matchNums, false);
    hasmatched.resize(matchNums, false);
}

bool MatchManager::matching_query(int matchID)
{
    return ismatching[matchID];
}

void MatchManager::matching_mark(int matchID)
{
    ismatching[matchID] = true;
}

bool MatchManager::matched_query(int matchID)
{
    return hasmatched[matchID];
}

void MatchManager::matched_mark(int matchID)
{
    hasmatched[matchID] = true;
}

void MatchManager::addCheckPair(int j, int i)
{
    checkPairs[i].push_back(j);
}

void MatchManager::addConnectScore(cv::Mat connectScore_)
{
    connectScore_.copyTo(connectScore);
}

void MatchManager::iPairIndex(int i, int &ind1, int &ind2)
{
    ind1 = pairIndex.at<float>(i, 0);
    ind2 = pairIndex.at<float>(i, 1);
}

int MatchManager::matchSize()
{
    return matchNums;
}

void MatchManager::pushMatch(cv::Mat match_cell, int i)
{
    matcheArr[i].push_back(match_cell);
}

void MatchManager::addLine3D(cv::Mat lineCell, int i)
{
    line3DArr[i].push_back(lineCell);
}

void MatchManager::addLine3DRevise(cv::Mat reviseCell, int i)
{
    line3DArr_revise[i].push_back(reviseCell);
}

void MatchManager::print(int i)
{
    std::cout << line3DArr[i] << std::endl;
    getchar();
}

float *MatchManager::line3DAllPtr()
{
    return line3DAll;
}

cv::Mat MatchManager::line3DMidPoint()
{
    cv::Mat line3DMidPt = cv::Mat(line3DNums, 3, CV_32FC1);

    float *line3DMidPtPtr = (float *)line3DMidPt.data;

#pragma omp parallel for
    for (int i = 0; i < line3DNums; i++)
    {
        float *curLine3D = line3DAll + i * LINE3DCOL;
        float *curMid = line3DMidPtPtr + i * 3;

        curMid[0] = (curLine3D[0] + curLine3D[3]) / 2;
        curMid[1] = (curLine3D[1] + curLine3D[4]) / 2;
        curMid[2] = (curLine3D[2] + curLine3D[5]) / 2;
    }

    return line3DMidPt;
}

void MatchManager::countLines()
{
    globalLineIndex.resize(matchNums);
    for (int i = 0; i < matchNums; i++)
    {
        globalLineIndex[i] = line3DNums;
        line3DNums = line3DNums + matcheArr[i].rows;
    }
}

int MatchManager::line3Dsize()
{
    return line3DNums;
}

cv::Mat *MatchManager::line3DArrPtr(int i)
{
    return &line3DArr[i];
}

float *MatchManager::line3DSingle(int pairID, int matchID)
{
    return (float *)line3DArr[pairID].data + line3DArr[pairID].cols * matchID;
}

cv::Mat *MatchManager::line3DRevisePtr(int i)
{
    return &line3DArr_revise[i];
}

std::vector<std::vector<int>> *MatchManager::checkPairsPtr()
{
    return &checkPairs;
}

cv::Mat *MatchManager::matcheArrPtr(int i)
{
    return &matcheArr[i];
}

ushort *MatchManager::matcheRowPtr(int pairID, int matchID)
{
    return (ushort *)matcheArr[pairID].data + matchID * matcheArr[pairID].cols;
}

std::vector<int> *MatchManager::globalSizePtr()
{
    return &globalLineIndex;
}

cv::Mat *MatchManager::allPairIndex()
{
    return &pairIndex;
}

cv::Mat *MatchManager::connectScoreM()
{
    return &connectScore;
}

int MatchManager::sweepHasMatched()
{
    int counter = 0;
    for (int i = 0; i < hasmatched.size(); i++)
        if (hasmatched[i] == true)
            counter++;

    return counter;
}
