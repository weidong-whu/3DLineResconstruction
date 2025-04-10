#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>  // C++17
#include <iostream>


template <typename T>
void write2txt(T* mat, int rows, int cols, std::string filename)
{
	std::ofstream writemat(filename, std::ios::out);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			writemat << mat[j] << " ";
		}
		writemat << std::endl;
		mat = mat + cols;
	}
	writemat.close();
}

void createOutputDirectory(const std::string& pathStr);

void saveMatBinary(const std::string& filename, const cv::Mat& mat);

cv::Mat loadMatBinary(const std::string& filename);
