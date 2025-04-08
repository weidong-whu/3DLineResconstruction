#pragma once
#pragma once

#include"SfMManager.h"
#include"MatchManager.h"
#include<queue>

struct SFM_INFO {
    std::vector<cv::Mat> camID;
    std::vector<cv::Mat> lineID;
    std::vector<cv::Mat> counters;
    cv::Mat pairID;
};

struct IMG_INFO {
    std::vector<cv::Mat> lines;
    std::vector<cv::Mat> cameras;
    std::vector<cv::Mat> centers;
};

struct ARR_INFO {
    cv::Mat meanArr;
    cv::Mat stdArr;
    double distmean;
};

struct SPACE_REC {
    cv::Mat lines3D;
    cv::Mat camid;
    cv::Mat lineid;
    cv::Mat counters;
    cv::Mat clusters;
};

struct PARAMS {
    //% optimization
    //params.maxAng = pi / 180 * 2;
    //params.projdis = 2;
    //params.dist3D = distmean;
    //params.colinearNum = 30;
    //params.coplanarNum = 30;
    //params.growAng = pi / 180;
    //params.longLineNum = 2;

    double maxAng = 3.141592653589793 / 180.0 * 2.0;
    double projdis = 2.0;
    double dist3D = 0.0;
    double colinearNum = 30;
    double coplanarNum = 30;
    double growAng = 3.141592653589793 / 180.0;
    double longLineNum = 2.0;

};



class MergeProcess
{
  
    int minimumCellCout = 10;
   
    SfMManager* sfmM;
    MatchManager* matchM;

    cv::Mat createLineMap(int imageid);
   
    void write2obj(std::string input_folder,
        std::string outname,
        float* l3ds, int rows);
 
    void lineSweepData(int ind, ushort** counter, ushort** camsID, ushort** lineID, float** error);

    
    std::vector<std::vector<std::vector<ushort>>> getCamID();
    std::vector<std::vector<std::vector<ushort>>> getLineID();
    
    ~MergeProcess()
    {
       
   
    }

  

    void sweepCheckLine(const ushort* limap,int imageid, int curmatch, float max_error, float mean, float std);
    
    public:
        std::vector<std::vector<std::vector<ushort>>>camsID;
        std::vector<std::vector<std::vector<ushort>>>lineID;
        std::vector<std::vector<std::vector<float>>>errors;

        MergeProcess(SfMManager* sfmM, MatchManager* matchM);
        void sweep4Image(int imageID, std::vector<int> matchids);
       
        void beginSweep();


        
};

