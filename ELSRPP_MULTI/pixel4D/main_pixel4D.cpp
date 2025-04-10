#include "callELSRPP.h"
#include <iostream>
#include <string>
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <time.h>
#include <vector>

// Define a very small constant L3D_EPS
#define L3D_EPS 1e-12

// Convert an Eigen matrix to an OpenCV 32-bit single-channel matrix
template <typename Derived> cv::Mat eigenToCvMat32FC1(const Eigen::MatrixBase<Derived> &mat)
{
    // Convert to float type
    Eigen::MatrixXf mat_float = mat.template cast<float>();

    cv::Mat result(mat_float.rows(), mat_float.cols(), CV_32FC1);
    // Copy row by row and column by column to ensure the order is consistent
    for (int i = 0; i < mat_float.rows(); ++i)
        for (int j = 0; j < mat_float.cols(); ++j)
            result.at<float>(i, j) = mat_float(i, j);

    return result;
}

// Helper function for point triangulation
Eigen::Vector3d linearHomTriangulation(std::list<std::pair<size_t, Eigen::Vector2d>> &obs,
                                       std::vector<Eigen::MatrixXd> &P)
{
    if (obs.size() == 0 || P.size() == 0)
        return Eigen::Vector3d(0, 0, 0);

    std::vector<Eigen::MatrixXd> Sx(obs.size(), Eigen::MatrixXd::Zero(2, 3));
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(obs.size() * 2, 4);
    std::list<std::pair<size_t, Eigen::Vector2d>>::iterator it = obs.begin();
    for (size_t i = 0; it != obs.end(); ++i, ++it)
    {
        Eigen::Vector2d pt = (*it).second;
        size_t camID = (*it).first;

        Sx[i](0, 1) = -1;
        Sx[i](0, 2) = pt.y();
        Sx[i](1, 0) = 1;
        Sx[i](1, 2) = -pt.x();

        A.block<2, 4>(i * 2, 0) = Sx[i] * P[camID];
    }

    Eigen::MatrixXd AtA(4, 4);
    AtA = A.transpose() * A;

    Eigen::MatrixXd U, V;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(AtA, Eigen::ComputeThinU | Eigen::ComputeThinV);

    U = svd.matrixU();
    V = svd.matrixV();

    Eigen::VectorXd X;

    X = V.col(3);
    X /= X(3);

    return Eigen::Vector3d(X(0), X(1), X(2));
}

// Read pixel data from Pix4D files
int read_pixel4d(SfMManager *sfm, MatchManager *match)
{

    std::string file1 = sfm->params_prefix + "calibrated_camera_parameters.txt";
    std::string file2 = sfm->params_prefix + "tp_pix4d.txt";

    boost::filesystem::path pf1(file1);
    boost::filesystem::path pf2(file2);

    // Check if Pix4D files exist
    if (!boost::filesystem::exists(pf1) || !boost::filesystem::exists(pf2))
    {
        std::cerr << "pix4d file '" << file1 << "' or '" << std::endl << file2 << "' does not exist!" << std::endl;
        return -1;
    }

    // Camera parameter file
    std::ifstream pix4d_cam_file;
    pix4d_cam_file.open(file1.c_str());

    std::string pix4d_cam_line;
    // Ignore descriptions...
    while (std::getline(pix4d_cam_file, pix4d_cam_line))
    {
        if (pix4d_cam_line.length() < 2)
            break;
    }

    std::vector<cv::Mat> Ms;
    std::vector<cv::Mat> Cs;
    std::vector<cv::Mat> cams;

    // Read camera data (sequentially)
    std::map<std::string, size_t> img2pos;
    std::map<size_t, std::string> pos2img;
    std::vector<std::string> cams_filenames;
    std::vector<Eigen::Matrix3d> cams_rotation;
    std::vector<Eigen::Matrix3d> cams_intrinsic;
    std::vector<Eigen::MatrixXd> cams_projection;
    std::vector<Eigen::Vector3d> cams_translation;
    std::vector<Eigen::Vector3d> cams_radial_dist;
    std::vector<Eigen::Vector2d> cams_tangential_dist;

    while (std::getline(pix4d_cam_file, pix4d_cam_line))
    {
        if (pix4d_cam_line.length() < 5)
            break;

        // Filename
        std::stringstream pix4d_stream(pix4d_cam_line);
        std::string filename, width, height;
        pix4d_stream >> filename >> width >> height;

        size_t lastindex = filename.find_last_of(".");
        std::string rawname = filename.substr(0, lastindex);

        img2pos[rawname] = cams_filenames.size();
        pos2img[cams_filenames.size()] = rawname;
        cams_filenames.push_back(filename);

        // Intrinsics
        Eigen::Matrix3d K;
        for (size_t i = 0; i < 3; ++i)
        {
            std::getline(pix4d_cam_file, pix4d_cam_line);
            pix4d_stream.clear();
            pix4d_stream.str(pix4d_cam_line);
            pix4d_stream >> K(i, 0) >> K(i, 1) >> K(i, 2);
        }
        cams_intrinsic.push_back(K);

        // Radial distortion
        Eigen::Vector3d radial;
        std::getline(pix4d_cam_file, pix4d_cam_line);
        pix4d_stream.clear();
        pix4d_stream.str(pix4d_cam_line);
        pix4d_stream >> radial(0) >> radial(1) >> radial(2);
        cams_radial_dist.push_back(radial);

        // Tangential distortion
        Eigen::Vector2d tangential;
        std::getline(pix4d_cam_file, pix4d_cam_line);
        pix4d_stream.clear();
        pix4d_stream.str(pix4d_cam_line);
        pix4d_stream >> tangential(0) >> tangential(1);
        cams_tangential_dist.push_back(tangential);

        // Translation
        Eigen::Vector3d t;
        std::getline(pix4d_cam_file, pix4d_cam_line);
        pix4d_stream.clear();
        pix4d_stream.str(pix4d_cam_line);
        pix4d_stream >> t(0) >> t(1) >> t(2);

        // Rotation
        Eigen::Matrix3d R;
        for (size_t i = 0; i < 3; ++i)
        {
            std::getline(pix4d_cam_file, pix4d_cam_line);
            pix4d_stream.clear();
            pix4d_stream.str(pix4d_cam_line);
            pix4d_stream >> R(i, 0) >> R(i, 1) >> R(i, 2);
        }

        cams_rotation.push_back(R);
        cv::Mat center = eigenToCvMat32FC1(t).t();

        sfm->addCamsCenter(center.clone());
        Cs.push_back(center.clone());

        t = -R * t;
        cams_translation.push_back(t);

        // Projection
        Eigen::MatrixXd P(3, 4);
        P.block<3, 3>(0, 0) = R;
        P.block<3, 1>(0, 3) = t;

        sfm->addCamsRT(eigenToCvMat32FC1(P));

        P = K * P;
        cams_projection.push_back(P);

        sfm->addImageNames(filename);

        cv::Mat Pmat = eigenToCvMat32FC1(P);

        cv::Mat subMat = Pmat.rowRange(0, 3).colRange(0, 3);
        Ms.push_back(subMat.clone().t());
        cams.push_back(Pmat.clone());
    }

    pix4d_cam_file.close();
    sfm->iniImageSize();
    sfm->iniCameraSize();

    for (int i = 0; i < cams.size(); i++)
    {
        sfm->addCamera(cams[i], i);
        int aw, ah, as;
        get_image_size_without_decode_image((sfm->inFolder + "/" + sfm->iImageNames(i)).c_str(), &aw, &ah, &as);
        sfm->addImSize(ah, aw, i);
    }

    // Read point data
    std::ifstream pix4d_point_file;
    pix4d_point_file.open(file2.c_str());

    std::string pix4d_point_line;

    std::map<std::string, std::list<unsigned int>> featuresPerCam;
    std::map<std::string, unsigned int> feat_key2id;
    std::map<unsigned int, std::string> feat_id2key;
    std::map<unsigned int, bool> feat_valid;
    std::map<unsigned int, Eigen::Vector3d> feat_pos3D;
    std::vector<std::list<std::pair<size_t, Eigen::Vector2d>>> feat_observations;

    std::string key;
    size_t key_img_pos;
    while (std::getline(pix4d_point_file, pix4d_point_line))
    {
        std::string id, rest;
        double px, py, scale;

        std::stringstream pix4d_stream(pix4d_point_line);
        pix4d_stream >> id >> rest;

        if (id.length() < 2)
            break;

        if (id.substr(0, 1) != "-")
        {
            if (rest.length() == 0)
            {
                // New key image
                key = id;
                key_img_pos = img2pos[key];
            }
            else
            {
                // New feature for current key image
                pix4d_stream.clear();

                pix4d_stream.str(pix4d_point_line);
                pix4d_stream >> id >> px >> py >> scale;

                // Check for new feature
                size_t fID;
                if (feat_key2id.find(id) == feat_key2id.end())
                {
                    // New feature
                    fID = feat_observations.size();
                    feat_key2id[id] = fID;
                    feat_id2key[fID] = id;
                    feat_valid[fID] = false;
                    feat_pos3D[fID] = Eigen::Vector3d(0, 0, 0);

                    feat_observations.push_back(std::list<std::pair<size_t, Eigen::Vector2d>>());
                }
                else
                {
                    // Existing feature
                    fID = feat_key2id[id];
                }

                // Add observation
                featuresPerCam[key].push_back(fID);
                feat_observations[fID].push_back(
                    std::pair<size_t, Eigen::Vector2d>(key_img_pos, Eigen::Vector2d(px, py)));
            }
        }
    }
    pix4d_point_file.close();

    std::cout << "Pix4D: #cameras = " << img2pos.size() << std::endl;
    std::cout << "Pix4D: #points  = " << feat_observations.size() << std::endl;

    // Triangulate points (parallel)
    std::cout << "triangulating..." << std::endl;
#ifdef L3DPP_OPENMP
#pragma omp parallel for
#endif // L3DPP_OPENMP
    for (int i = 0; i < feat_observations.size(); ++i)
    {
        std::list<std::pair<size_t, Eigen::Vector2d>> obs = feat_observations[i];
        if (obs.size() > 2)
        {
            Eigen::Vector3d P = linearHomTriangulation(obs, cams_projection);

            if (P.norm() > L3D_EPS)
            {
                feat_valid[i] = true;
                feat_pos3D[i] = P;
            }
        }
    }

    std::cout << "prepare for elsrpp..." << std::endl;
    float pt3d[3];
    std::vector<point_info> point_cluster;
    std::vector<uint> cam_IDs;
    sfm->initialImagePoints();

    cv::Mat mscores = cv::Mat::zeros(sfm->camsNumber(), sfm->camsNumber(), CV_16UC1);

    std::vector<cv::Mat *> errs(feat_observations.size());
    for (int i = 0; i < feat_observations.size(); i++)
        errs[i] = new cv::Mat;

    for (int i = 0; i < feat_observations.size(); ++i)
    {

        cv::Mat pos3D(1, 3, CV_32FC1);
        pos3D.at<float>(0, 0) = feat_pos3D[i][0];
        pos3D.at<float>(0, 1) = feat_pos3D[i][1];
        pos3D.at<float>(0, 2) = feat_pos3D[i][2];

        sfm->add_points_space3D(pos3D);

        if (feat_valid[i] != true)
            continue;

        cv::Mat points;
        std::list<std::pair<size_t, Eigen::Vector2d>> obs = feat_observations[i];
        std::list<std::pair<size_t, Eigen::Vector2d>>::iterator it = obs.begin();
        for (; it != obs.end(); ++it)
        {
            Eigen::Vector2d pt = (*it).second;
            size_t camID = (*it).first;

            cam_IDs.push_back(camID);

            pos3D.at<float>(0, 0) = pt[0];
            pos3D.at<float>(0, 1) = pt[1];
            pos3D.at<float>(0, 2) = i + 1;

            int xsize, ysize;
            sfm->iImSize(camID, ysize, xsize);
            cv::Mat subline = (cv::Mat_<float>(1, 3) << camID, pt[0], pt[1]);
            points.push_back(subline.clone());

            sfm->addImagePoints(camID, pos3D);

            point_cluster.push_back(point_info{pos3D.at<float>(0, 0), pos3D.at<float>(0, 1), (uint)camID});
        }

        sfm->multiPoints.push_back(points.clone());

        pt3d[0] = feat_pos3D[i][0];
        pt3d[1] = feat_pos3D[i][1];
        pt3d[2] = feat_pos3D[i][2];

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
    // write2txt((ushort*)mscores.data, mscores.rows, mscores.cols, sfm->inputFolder() + "\\scores.txt");

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

    return 1;
}

int main(int argc, char *argv[])
{
#pragma region TCLAP cmd line trans

    // Introduction of the program
    TCLAP::CmdLine cmd("ELSRPP");
    // Folder containing the images
    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", true, ".", "string");
    cmd.add(inputArg);

    // Folder for output
    TCLAP::ValueArg<std::string> outputArg("o", "out_folder", "folder for output", true, ".", "string");
    cmd.add(outputArg);

    // Folder containing the project files <project_prefix>_calibrated_camera_parameters.txt and
    // <project_prefix>_tp_pix4d.txt
    TCLAP::ValueArg<std::string> pix4dArg(
        "b", "params_folder",
        "folder containing the project files <project_prefix>_calibrated_camera_parameters.txt and "
        "<project_prefix>_tp_pix4d.txt",
        true, "", "string");
    cmd.add(pix4dArg);
    // Project name and output file prefix
    TCLAP::ValueArg<std::string> prefixArg("f", "project_prefix", "project name and output file prefix", true, "",
                                           "string");
    cmd.add(prefixArg);
    // Line extraction method
    TCLAP::ValueArg<std::string> lineExtMethod("l", "line_ext_method",
                                               "line extraction method, if designate this command,"
                                               " the line file should exist in the same folder with image files, and "
                                               "the naming rule of line file should follow:\n"
                                               "-l lsd\n"
                                               "-l ag3line\n"
                                               "-l edline\n",
                                               false, "lsd", "string");
    cmd.add(lineExtMethod);
    // The maximum size of input image
    TCLAP::ValueArg<int> maxSize("s", "max_image_size", "the maximum size of input image", false, 99999, "int");
    cmd.add(maxSize);
    // The maximum line number of input lines
    TCLAP::ValueArg<int> max_LineNum("n", "max_line_num", "the maximum line number of input lines", false, 99999,
                                     "int");
    cmd.add(max_LineNum);
    cmd.parse(argc, argv);
#pragma endregion

    clock_t start, end;
    start = clock();

    int maxwidth = maxSize.getValue();
    int maxLineNum = max_LineNum.getValue();
    std::string lineType = lineExtMethod.getValue();
    std::string imageFolder = inputArg.getValue().c_str();
    std::string sfmFolder = pix4dArg.getValue().c_str();
    std::string projextPrefix = prefixArg.getValue().c_str();
    std::string outFolder = outputArg.getValue().c_str();

    // Check if parameter files exist
    std::string params_prefix = sfmFolder + "/" + projextPrefix;
    if (params_prefix.substr(params_prefix.length() - 1, 1) != "_")
        params_prefix += "_";

    SfMManager sfm(params_prefix, imageFolder, outFolder);
    MatchManager match;

    // Read sfm data
    read_pixel4d(&sfm, &match);

    elsrpp(&sfm, &match, lineType, maxwidth, maxLineNum);

    return 0;
}
