#include "callELSRPP.h"
#include <iostream>
#include <string>
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <time.h>
#include <vector>

int read_VisualSfM(SfMManager *sfm, MatchManager *match)
{

    std::string inputFolder = sfm->inputFolder();
    std::string nvmFile = sfm->nvmFile();
    // check if NVM file exists
    if (!std::filesystem::exists(nvmFile))
    {
        std::cerr << "No  NVM file!" << std::endl;
        return -1;
    }

    // read NVM file
    std::ifstream nvm_file;
    nvm_file.open(nvmFile.c_str());

    std::string nvm_line;
    std::getline(nvm_file, nvm_line); // ignore first line...
    std::getline(nvm_file, nvm_line); // ignore second line...

    // read number of images
    std::getline(nvm_file, nvm_line);
    std::stringstream nvm_stream(nvm_line);
    unsigned int num_cams;
    nvm_stream >> num_cams;

    std::vector<point_info> point_cluster;

    if (num_cams == 0)
    {
        std::cerr << "No aligned cameras in NVM file!" << std::endl;
        return -1;
    }

    // read camera data (sequentially)
    cv::Mat imidx_Mf;

    cv::Mat R(3, 3, CV_32FC1);
    cv::Mat C(3, 1, CV_32FC1);

    std::vector<cv::Mat> Ms;
    std::vector<cv::Mat> Cs;
    std::vector<cv::Mat> cams;

    std::vector<cv::Mat *> errs(num_cams);
    for (int i = 0; i < num_cams; i++)
        errs[i] = new cv::Mat;

    for (unsigned int i = 0; i < num_cams; ++i)
    {
        std::getline(nvm_file, nvm_line);

        // image filename
        std::string filename;

        // focal_length,quaternion,center,distortion
        double focal_length, qx, qy, qz, qw;
        double Cx, Cy, Cz, dist;

        nvm_stream.str("");
        nvm_stream.clear();
        nvm_stream.str(nvm_line);
        nvm_stream >> filename >> focal_length >> qw >> qx >> qy >> qz;
        nvm_stream >> Cx >> Cy >> Cz >> dist;

        sfm->addImageNames(filename);
        sfm->addCamsFocals(focal_length);

        // rotation amd translation
        R.at<float>(0, 0) = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz;
        R.at<float>(0, 1) = 2.0 * qx * qy - 2.0 * qz * qw;
        R.at<float>(0, 2) = 2.0 * qx * qz + 2.0 * qy * qw;

        R.at<float>(1, 0) = 2.0 * qx * qy + 2.0 * qz * qw;
        R.at<float>(1, 1) = 1.0 - 2.0 * qx * qx - 2.0 * qz * qz;
        R.at<float>(1, 2) = 2.0 * qy * qz - 2.0 * qx * qw;

        R.at<float>(2, 0) = 2.0 * qx * qz - 2.0 * qy * qw;
        R.at<float>(2, 1) = 2.0 * qy * qz + 2.0 * qx * qw;
        R.at<float>(2, 2) = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy;

        C.at<float>(0, 0) = Cx;
        C.at<float>(1, 0) = Cy;
        C.at<float>(2, 0) = Cz;

        cv::Mat t = -R * C;

        cv::Mat Rt;
        cv::hconcat(R, t, Rt);

        sfm->addCamsRT(Rt);
        sfm->addCamsCenter(C.t());

        // for error construction
        cv::Mat KK = cv::Mat::zeros(3, 3, CV_32FC1);
        KK.at<float>(0, 0) = focal_length;
        KK.at<float>(1, 1) = focal_length;
        KK.at<float>(2, 2) = 1;

        int aw, ah, as;
        get_image_size_without_decode_image((inputFolder + "/" + filename).c_str(), &aw, &ah, &as);

        KK.at<float>(0, 2) = aw / 2.0;
        KK.at<float>(1, 2) = ah / 2.0;

        Cs.push_back(C.t());
        cv::Mat cam = KK * Rt;
        Ms.push_back(cam.rowRange(0, 3).colRange(0, 3).t());
        cams.push_back(cam);
    }

    std::vector<float> xsize;
    std::vector<float> ysize;

    sfm->iniImageSize();
    sfm->iniCameraSize();
    // read image size
    for (int i = 0; i < num_cams; i++)
    {

        printf("read image %d\n", i);

        int aw, ah, as;
        get_image_size_without_decode_image((inputFolder + "/" + sfm->iImageNames(i)).c_str(), &aw, &ah, &as);

        xsize.push_back((float)aw / 2.0);
        ysize.push_back((float)ah / 2.0);

        sfm->addImSize(ah, aw, i);
        sfm->addCameraBySize(ah, aw, i);
    }

    // std::ofstream ofs;
    // ofs.open(inputFolder + "//output//multiPoints.txt", std::ios::out);

    // read number of images
    std::getline(nvm_file, nvm_line); // ignore line...
    std::getline(nvm_file, nvm_line);
    nvm_stream.str("");
    nvm_stream.clear();
    nvm_stream.str(nvm_line);
    unsigned int num_points;
    nvm_stream >> num_points;

    // read features (for image similarity calculation)
    cv::Mat pos3D(1, 3, CV_32FC1);

    std::vector<cv::Mat> points2D_N(num_cams);
    cv::Mat mscores = cv::Mat::zeros(num_cams, num_cams, CV_16UC1);

    std::vector<uint> cam_IDs;
    sfm->initialImagePoints();
    float depth;
    cv::Mat depthes;

    float pt3d[3];

    for (unsigned int i = 0; i < num_points; ++i)
    {
        // 3D position
        std::getline(nvm_file, nvm_line);
        std::istringstream iss_point3D(nvm_line);
        double px, py, pz, colR, colG, colB;
        iss_point3D >> px >> py >> pz;
        iss_point3D >> colR >> colG >> colB;

        pos3D.at<float>(0, 0) = px;
        pos3D.at<float>(0, 1) = py;
        pos3D.at<float>(0, 2) = pz;

        sfm->add_points_space3D(pos3D);
        //  num of views for each 3D points
        unsigned int num_views;
        iss_point3D >> num_views;

        unsigned int camID, siftID;
        float posX, posY;

        cv::Mat points;
        for (unsigned int j = 0; j < num_views; ++j)
        {
            iss_point3D >> camID >> siftID;
            iss_point3D >> posX >> posY;

            cam_IDs.push_back(camID);

            pos3D.at<float>(0, 0) = posX;
            pos3D.at<float>(0, 1) = posY;
            pos3D.at<float>(0, 2) = i + 1;

            cv::Mat subline = (cv::Mat_<float>(1, 3) << camID, posX + xsize[camID], posY + ysize[camID]);
            points.push_back(subline);

            sfm->addImagePoints(camID, pos3D);

            float *CM = sfm->iCameraMatPtr(camID);

            sfm->allPoints3D()->at<float>();

            point_cluster.push_back(point_info{posX, posY, camID});
        }

        // ofs << points << std::endl;
        sfm->multiPoints.push_back(points);

        pt3d[0] = px;
        pt3d[1] = py;
        pt3d[2] = pz;

        point2lineerr(pt3d, point_cluster, Ms, Cs, errs);

        point_cluster.clear();

        for (int ii = 0; ii < cam_IDs.size(); ii++)
            for (int jj = ii + 1; jj < cam_IDs.size(); jj++)
            {
                mscores.at<ushort>(cam_IDs[ii], cam_IDs[jj])++;
                mscores.at<ushort>(cam_IDs[jj], cam_IDs[ii])++;
            }

        cam_IDs.clear();
    }

    // ofs.close();
    int nbins = sfm->bins;

    cv::Mat generalized_err;
    cv::Mat mean_std(errs.size(), 2, CV_32FC1);
    cv::Mat max_error(errs.size(), 1, CV_32FC1);
    for (int mm = 0; mm < errs.size(); mm++)
    {
        cv::Mat err_sub = *errs[mm];

        if (err_sub.rows < 5)
            continue;

        cv::Mat std_, mean_;
        cv::meanStdDev(err_sub, mean_, std_);

        float mmax_thre = mean_.at<double>(0, 0) + 3 * std_.at<double>(0, 0);
        float mmin_thre = mean_.at<double>(0, 0) - 3 * std_.at<double>(0, 0);

        cv::Mat new_sub_err;
        float maxv = 0;
        for (int i = 0; i < err_sub.rows; i++)
        {
            if (err_sub.at<float>(i, 0) > mmax_thre || err_sub.at<float>(i, 0) < mmin_thre)
                continue;
            new_sub_err.push_back(err_sub.at<float>(i, 0));

            if (abs(err_sub.at<float>(i, 0)) > maxv)
                maxv = abs(err_sub.at<float>(i, 0));
        }

        cv::meanStdDev(new_sub_err, mean_, std_);
        float mean_new = mean_.at<double>(0, 0);
        float std_new = std_.at<double>(0, 0);

        mean_std.at<float>(mm, 0) = mean_new;
        mean_std.at<float>(mm, 1) = std_new;

        max_error.at<float>(mm, 0) = maxv;

        new_sub_err = (new_sub_err - mean_new) / std_new;

        generalized_err.push_back(new_sub_err);
    }

    std::vector<uint> counts(nbins, 0);
    float interval = 3.0 / nbins;
    for (int i = 0; i < generalized_err.rows; i++)
    {
        int ibin = std::floor(std::abs(generalized_err.at<float>(i, 0)) / interval);
        if (ibin >= nbins)
            ibin = nbins - 1;
        if (ibin < 0)
            continue;
        counts[ibin] = counts[ibin] + 1;
    }

    cv::Mat prior = cv::Mat::zeros(nbins, nbins, CV_32FC1);

    for (int i = 0; i < nbins; i++)
    {
        float sum = 0;

        for (int j = 0; j < i; j++)
            sum = sum + counts[j];

        for (int j = 0; j < i; j++)
            prior.at<float>(i, j) = counts[j] / sum;

        float sub = nbins - i;
        for (int j = i; j < nbins; j++)
            prior.at<float>(i, j) = 1.0 / sub;
    }

    sfm->addTrainPrioi(prior, mean_std, max_error);
    sfm->interval = interval;

    match->addConnectScore(mscores);

    nvm_file.close();

    return 1;
}

int main(int argc, char *argv[])
{

#pragma region TCLAP cmd line trans

    TCLAP::CmdLine cmd("ELSRPP"); // introduction of the program
    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", true, ".", "string");
    cmd.add(inputArg);
    TCLAP::ValueArg<std::string> nvmArg("m", "nvmfile", "nvmfile from visual sfm", true, ".", "string");
    cmd.add(nvmArg);

    TCLAP::ValueArg<std::string> outputArg("o", "out_folder", "folder containing the images", true, ".", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<std::string> lineExtMethod("l", "line_ext_method",
                                               "line extraction method, if designate this command,"
                                               " the line file should exist in the same folder with image files, and "
                                               "the naming rule of line file should follow:\n"
                                               "-l lsd\n"
                                               "-l ag3line\n"
                                               "-l edline\n",
                                               false, "lsd", "string");
    cmd.add(lineExtMethod);
    TCLAP::ValueArg<int> maxSize("s", "max_image_size", "the maximum size of input image", false, 99999, "int");
    cmd.add(maxSize);
    TCLAP::ValueArg<int> max_LineNum("n", "max_line_num", "the maximum line number of input lines", false, 99999,
                                     "int");
    cmd.add(max_LineNum);
    cmd.parse(argc, argv);
#pragma endregion

    clock_t start, end;
    start = clock();

    std::string inputFolder = inputArg.getValue().c_str();
    std::string outFolder = outputArg.getValue().c_str();
    std::string nvmFile = nvmArg.getValue().c_str();
    std::string lineType = lineExtMethod.getValue();
    int maxwidth = maxSize.getValue();
    int maxLineNum = max_LineNum.getValue();

    SfMManager sfm(nvmFile, inputFolder, outFolder);
    MatchManager match;

    // read .nvm of VisualSfM
    read_VisualSfM(&sfm, &match);
    std::cout << "process visual sfm in " << (float)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

    elsrpp(&sfm, &match, lineType, maxwidth, maxLineNum);

    return 0;
}
