#include "line_detector/ag3line/RegManager.h"


RegManager::RegManager(int m,int n)
{
	reg = (cv::Point3i *) calloc( (size_t) (m+n), sizeof(cv::Point) );
	iteraarr=(uchar *) calloc( (size_t) (m+n), sizeof(uchar) );
	rectreg= (cv::Point3i *) calloc( (size_t) (m+n)*10, sizeof(cv::Point) );
	anchorreg= (cv::Point3i *) calloc( (size_t) (m+n), sizeof(cv::Point) );
	gradarr=(float *) calloc( (size_t) (m+n), sizeof(float) );

	s_x2 = 0;
	s_xy = 0;
	s_x = 0;
	s_y2 = 0;
	s_y = 0;
}


RegManager::~RegManager(void)
{
	free((cv::Point3i*)reg);
	free((uchar*)iteraarr);
	free((cv::Point3i*)rectreg);
	free((cv::Point3i*)anchorreg);
	free((float*)gradarr);
}

int RegManager:: splitTheReg()
{
	maxitera=0;
	for( i=ed1;i<ed2;i++)
	{
		if (iteraarr[i]>maxitera)
		{
			maxidx=i;
			maxitera=iteraarr[i];
		}
	}
	for(i=ed3;i<ed4;i++)
	{
		if (iteraarr[i]>maxitera)
		{
			maxidx=i;
			maxitera=iteraarr[i];
		}
	}
	if(maxidx==0)
		return 0;
	//split via the maxidx
	//1st maxid falls between ed1 and ed2
	if (maxidx<=ed2&&maxidx>=ed1)
	{
		if (ed2-maxidx>ed4-ed3+maxidx-ed1)
		{
			ed1=maxidx;
			iteraarr[ed1]=1;
			ed3=ed4;
		}
		else 
		{				
			ed2=maxidx-1;
			
		}
	}
	//2st maxid falls between ed3 and ed4
	else
	{
		if (ed4-maxidx>ed2-ed1+maxidx-ed3)
		{
			ed3=maxidx;
			iteraarr[ed3]=1;
			ed1=ed2;
		}
		else 
		{
			ed4=maxidx-1;
		}
	}
	counter=ed4-ed3+ed2-ed1;
	return 1;
}

bool RegManager::satisfyGradAligned(int anchorsize,int idx,float dis_2)
{
	float mindis;
	for(j= 0;j<anchorsize;j++)
	{
		if(( rectreg[idx].y-anchorreg[j].y)*(rectreg[idx].y-anchorreg[j].y)+
			(rectreg[idx].x-anchorreg[j].x)*(rectreg[idx].x-anchorreg[j].x)<=dis_2)
			return true;
	}

	
	
	return false;
}

void RegManager::revise_sumxy(float x, float y)
{
	s_x2 += x * x;
	s_y2 += y * y;
	s_xy += x * y;
	s_x += x;
	s_y += y;
}
