#pragma once

#include"MatchManager.h"
#include"SfMManager.h"
class SamplingManager
{

struct mcell
{
	ushort imageid;
	ushort lineID;
};

public:
	SamplingManager(SfMManager* sfmM);
	~SamplingManager();

private:
	std::vector<std::vector<mcell>> line_line_m;
};


