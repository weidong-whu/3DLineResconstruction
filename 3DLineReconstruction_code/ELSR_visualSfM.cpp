//std
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <ctime>
#include <sys/stat.h>
#include <thread>
//mine
#include "SingleImage.h"
#include "PairMatch.h"
#include "LineSweep.h"
#include "Parameters.h"
#include "ThreadPool.h"
#include "read_VisualSfM.h"
#include "IO.h"
#include "Epipolar.h"
#include "BasicMath.h"
#include "LineCluster.h"

#include <fstream>
#include <windows.h>
#include <time.h>
#include <filesystem>
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
//marker



int main(int argc, char* argv[])
{
#pragma region TCLAP cmd line trans
	TCLAP::CmdLine cmd("ELSRPP"); //introduction of the program

	TCLAP::ValueArg<std::string> inputFld("i", "input_folder", "folder containing the images", true, "", "string");
	cmd.add(inputFld);

	TCLAP::ValueArg<int> lineExtMethod("l", "line_ext_method", "line extraction method, if designate this command,"
		" the line file should exist in the same folder with image files, and the naming rule of line file should follow:\n"
		"-l 0 auto detect line with opencv lsd\n"
		"-l 1 line file with extension .lsd\n"
		"-l 2 line file with extension .deeplsd\n"
		"-l 3 line file with extension .edline\n"
		"if not designate this command or with -1, the line_files_folder and line_files_extension must not be null", false, 4, "int");
	cmd.add(lineExtMethod);

	TCLAP::ValueArg<std::string> lineFileFolder("f", "line_files_folder", "folder save lines extraction from other method ", false, "", "string");
	cmd.add(lineFileFolder);

	TCLAP::ValueArg<std::string> lineFileExtension("e", "line_files_extension", "line file extension, which should be same with image file but"
		" not include the image extension\n, example:\n image file name: A.jpg\n line file name: A.jpg.lines\n this command: -e .jpg.lines",
		false, ".deeplsd", "string");
	cmd.add(lineFileExtension);

	TCLAP::ValueArg<bool> fromColmap("c", "from_colmap", "use colmap as input data, 0 means not use colmap", false, false, "bool");
	cmd.add(fromColmap);

	TCLAP::ValueArg<int> maxSize("s", "max_image_size", "the maximum size of input image", false, 99999, "int");
	cmd.add(maxSize);

	TCLAP::ValueArg<int> max_LineNum("n", "max_line_num", "the maximum line number of input lines", false, 99999, "int");
	cmd.add(max_LineNum);

	cmd.parse(argc, argv);

#pragma endregion

	clock_t start, end;
	start = clock();

	
	std::string inputFolder = inputFld.getValue().c_str();
	std::string nvmFile = "res.nvm";
	nvmFile = "\\" + nvmFile;

	int uselsd = lineExtMethod.getValue();
	int fromcolmap = fromColmap.getValue();
	int maxwidth = maxSize.getValue();
	int maxLineNum = max_LineNum.getValue();
	std::string lineFolder = lineFileFolder.getValue().c_str();
	std::string lineExt = lineFileExtension.getValue().c_str();

	if (uselsd == -1 && (lineFolder == "" || lineExt == "")) {
		std::cout << "Error:  line_ext_method or line_files_folder and line_files_extension should be designate at least one of them!\n use --help for more help" << std::endl;
		return -1;
	}

	printf("uselsd %d fromcolmap %d maxwidth %d\n", uselsd, fromcolmap, maxwidth);

	//if (!std::filesystem::exists(inputFolder + "\\output"));
	//	std::filesystem::create_directories(inputFolder + "\\output");
	
	std::cout << inputFolder << nvmFile << std::endl;
	

	int knn_image =10;
	int conectionNum = 50;

	SfMManager sfm(inputFolder, nvmFile,ADAPTIVE_BINS);
	MatchManager match;

	// read .nvm of VisualSf Mread_VisualSfM
	read_VisualSfM(&sfm, &match, knn_image, conectionNum, fromcolmap);

	std::cout << "process visual sfm in "
	<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

	//getchar();
	//1 Process single images,run_old_stream in multiple threads 
	start = clock();
	int* nums = new int[sfm.camsNumber()];

	#pragma omp parallel for
	for (int i = 0; i < sfm.camsNumber(); i++)
		nums[i] = processImage(&sfm, i, INTERSECT_COS, INTERSECT_DIS, SUPPORT_POINT_NUM, maxwidth, uselsd,maxLineNum, lineFolder, lineExt);
	
	//match index ana
	std::vector <int> linesize;
	sfm.lineSize(linesize);
	match.initializeM(linesize, knn_image, conectionNum);

	std::cout << "processImage sfm in "
		<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

	start = clock();
	int matchIM1, matchIM2;
	#pragma omp parallel for
	for (int matchID = 0; matchID < match.matchSize(); matchID++)
	{
		match.iPairIndex(matchID, matchIM1, matchIM2);
		matchPair(&sfm, &match, matchID,POINT_LINE_DIS, LINE_LINE_ANG);
	}

	std::cout << "matchPair in "
		<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

	start = clock();
	MergeProcess* mp= new MergeProcess(&sfm, &match);
	mp->beginSweep();

	std::cout << "sweep in in "
		<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

	//Line Cluster
	start = clock();
	
	lineCluster(&sfm, mp, inputFolder);

	std::cout << "line cluster in in "
		<< (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

	return 0;
}

