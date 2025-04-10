#include "ReadLoad.h"

namespace fs = std::filesystem;

void createOutputDirectory(const std::string &pathStr)
{
    fs::path dir(pathStr);

    try
    {
        if (!fs::exists(dir))
        {
            fs::create_directories(dir); //
            std::cout << "Directory created: " << dir << std::endl;
        }
        else
        {
            std::cout << "Directory already exists: " << dir << std::endl;
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

void saveMatBinary(const std::string &filename, const cv::Mat &mat)
{
    std::ofstream ofs(filename, std::ios::binary);

    int type = mat.type();
    int rows = mat.rows;
    int cols = mat.cols;

    ofs.write((char *)&rows, sizeof(int));
    ofs.write((char *)&cols, sizeof(int));
    ofs.write((char *)&type, sizeof(int));
    ofs.write((char *)mat.data, mat.total() * mat.elemSize());

    ofs.close();
}

cv::Mat loadMatBinary(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);

    int rows, cols, type;
    ifs.read((char *)&rows, sizeof(int));
    ifs.read((char *)&cols, sizeof(int));
    ifs.read((char *)&type, sizeof(int));

    cv::Mat mat(rows, cols, type);
    ifs.read((char *)mat.data, mat.total() * mat.elemSize());

    ifs.close();
    return mat;
}
