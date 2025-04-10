#include "callELSRPP.h"

void elsrpp(SfMManager *sfm, MatchManager *match, std::string lineType, int maxwidth, int maxLineNum)
{
    createOutputDirectory(sfm->outFolder + "\\" + lineType + "\\");

    clock_t start, end;

    // 1 Process single images,run_old_stream in multiple threads
    start = clock();
#pragma omp parallel for
    for (int i = 0; i < sfm->camsNumber(); i++)
        processImage(sfm, i, INTERSECT_COS, INTERSECT_DIS, SUPPORT_POINT_NUM, maxwidth, lineType, maxLineNum);

    // match index ana
    std::vector<int> linesize;
    sfm->lineSize(linesize);
    match->initializeM(linesize, sfm->knn_image, sfm->conectionNum);

    std::cout << "processImage sfm in " << (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();
    int matchIM1, matchIM2;
#pragma omp parallel for
    for (int matchID = 0; matchID < match->matchSize(); matchID++)
    {
        match->iPairIndex(matchID, matchIM1, matchIM2);
        matchPair(sfm, match, matchID, POINT_LINE_DIS, LINE_LINE_ANG);
    }

    std::cout << "matchPair in " << (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();
    MergeProcess *mp = new MergeProcess(sfm, match);
    mp->beginSweep();

    std::cout << "sweep in in " << (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

    // Line Cluster
    start = clock();
    lineCluster(sfm, mp);

    std::cout << "line cluster in in " << (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
}
