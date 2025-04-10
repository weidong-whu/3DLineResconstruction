#pragma once
#include<vector>
#include <opencv2/core.hpp>

#define LINE3DCOL 6

class MatchManager
{
public:
	
	~MatchManager()
	{
		delete[] line3DAll;
		delete[] matchesAll;
	}

	bool matching_query(int matchID);
	void matching_mark(int matchID);

	bool matched_query(int matchID);
	void matched_mark(int matchID);



	void iPairIndex(int i, int& ind1, int& ind2);
	cv::Mat* allPairIndex();
	
	int  matchSize();
	void pushMatch(cv::Mat match_cell, int i);
	void addLine3D(cv::Mat lineCell, int i);
	void addLine3DRevise(cv::Mat reviseCell, int i);

	void print(int i);

	void countLines();
	cv::Mat line3DMidPoint();
	int line3Dsize();

	
	void image2imageScore(int im1, int im2, int& score);
	
	std::vector<cv::Mat>* line3DArrPtr();
	std::vector<cv::Mat>* matcheArrPtr();
	std::vector<int >* globalSizePtr();

	void mergeLinesandMatches();
	
	float* line3DAllPtr();

	void addConnectScore(cv::Mat connectScore);

	std::vector < std::vector<int>>*checkPairsPtr();
	cv::Mat* connectScoreM();

	void connectImages(int imageID,std::vector<int>& connectPairID);

	void addCheckPair(int j, int i);

	cv::Mat* matcheArrPtr(int i);
	cv::Mat* line3DArrPtr(int i);
	cv::Mat* line3DRevisePtr(int i);
	float* line3DSingle(int pairID,int matchID);

	ushort* matcheRowPtr(int pairID, int matchID);

	int sweepHasMatched();
	
	void initializeM(std::vector<int>& lineSize, int knn, int minconnect);

	void writePairs(std::string addr);

private:
	std::vector<cv::Mat> line3DArr;
	std::vector<cv::Mat> line3DArr_revise;
	std::vector<cv::Mat> matcheArr;
	std::vector<int> globalLineIndex;

	std::vector<bool>ismatching;
	std::vector<bool>hasmatched;



	std::vector<std::vector<int>>checkPairs;

	cv::Mat connectScore;


	cv::Mat pairIndex;
	
	int matchNums=0;
	int line3DNums=0;
	short* matchesAll = 0;
	float* line3DAll = 0;

	
};



