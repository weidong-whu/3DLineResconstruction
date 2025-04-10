#include "SingleImage.h"
#include "PairMatch.h"
#include "LineSweep.h"
#include "Parameters.h"
#include <ctime>
#include "sfm_analysis.h"

#include "BasicMath.h"
#include "LineCluster.h"

void elsrpp(SfMManager* sfm, MatchManager* match, std::string lineType, int maxwidth, int maxLineNum);